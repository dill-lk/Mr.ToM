"""
rmoe/eval.py — Benchmark Runner & Evaluation Harness.

Implements the full evaluation pipeline from paper §4:
  • Load a CSV benchmark dataset (image_path, ground_truth, organ, modality, …)
  • Run the R-MoE pipeline on every case (mock or live)
  • Compute per-case and aggregate metrics:
      – Top-1 / Top-3 accuracy
      – Precision, Recall, F1-Score
      – AUC (macro one-vs-rest via trapezoidal rule)
      – ECE  (Expected Calibration Error, 10 bins)
      – Brier score
      – Type-I error rate  (false positive rate, FPR)
      – Type-II error rate (false negative rate, FNR = 1 − Recall)
      – Escalation rate    (% cases escalated to human)
      – Mean recursive iterations
      – Mean final Sc and σ²
      – Mean inference time (seconds)
  • Print ASCII results table, per-case trace, and overall metric table
  • Produce a LaTeX tabular block for direct paper inclusion
  • Save JSON results file

Paper Table 1 baselines (reproduced for comparison):
  R-MoE   F1=0.92  TypeI=5.2%  ECE=0.08  t=45 s
  GPT-4V  F1=0.85  TypeI=7.8%  ECE=0.15  t=32 s
  Gemini  F1=0.87  TypeI=7.1%  ECE=0.13  t=38 s
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkCase:
    """One entry in the benchmark dataset CSV."""
    case_id: str
    image_path: str
    ground_truth: str           # canonical diagnosis label
    ground_truth_icd11: str     # expected ICD-11 code
    organ: str                  # lung | thyroid | breast | liver | prostate | neuro | msk
    modality: str               # CXR | CT | MRI | US | PET
    expected_risk_score: str    # e.g. "Lung-RADS 4X"
    notes: str = ""
    prior_image_path: str = ""  # optional prior scan for temporal test


@dataclass
class CaseResult:
    """Result of running the R-MoE pipeline on one BenchmarkCase."""
    case: BenchmarkCase
    predicted_diagnosis: str    = ""
    predicted_probability: float = 0.0
    sc: float                   = 0.0
    sigma2: float               = 0.0
    iterations: int             = 0
    elapsed_s: float            = 0.0
    escalated: bool             = False
    icd11_predicted: str        = ""
    risk_score_predicted: str   = ""

    # Derived
    top1_correct: bool          = False   # predicted == ground_truth (normalised)
    top3_correct: bool          = False   # ground_truth in top-3 DDx
    icd11_correct: bool         = False
    all_ddx: List[Dict]         = field(default_factory=list)
    error: str                  = ""      # non-empty if pipeline crashed


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics across all benchmark cases."""
    n_cases: int             = 0

    # Classification
    accuracy:       float    = 0.0   # top-1
    top3_accuracy:  float    = 0.0
    precision:      float    = 0.0
    recall:         float    = 0.0
    f1:             float    = 0.0
    auc:            float    = 0.0   # macro one-vs-rest approximation

    # Calibration
    ece:            float    = 0.0
    brier:          float    = 0.0

    # Error rates
    type1_error:    float    = 0.0   # FPR  (false malignancy calls)
    type2_error:    float    = 0.0   # FNR  (missed diagnoses)

    # Pipeline behaviour
    escalation_rate:   float = 0.0
    mean_iterations:   float = 0.0
    mean_sc:           float = 0.0
    mean_sigma2:       float = 0.0
    mean_elapsed_s:    float = 0.0

    # ICD coding
    icd11_accuracy:    float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Built-in synthetic benchmark dataset
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_CASES: List[Dict] = [
    # ── Chest ─────────────────────────────────────────────────────────────────
    {"case_id": "MIMIC-CXR-001", "image_path": "data/images/cxr_001.png",
     "ground_truth": "Pulmonary adenocarcinoma",   "ground_truth_icd11": "2C25.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "Lung-RADS 4X",
     "notes": "3.2cm spiculated LUL mass, corona radiata"},
    {"case_id": "MIMIC-CXR-002", "image_path": "data/images/cxr_002.png",
     "ground_truth": "Community-acquired pneumonia", "ground_truth_icd11": "CA40.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "RLL lobar consolidation, air bronchogram"},
    {"case_id": "MIMIC-CXR-003", "image_path": "data/images/cxr_003.png",
     "ground_truth": "Pleural effusion",            "ground_truth_icd11": "CB21.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "Right costophrenic blunting ~200ml"},
    {"case_id": "MIMIC-CXR-004", "image_path": "data/images/cxr_004.png",
     "ground_truth": "Pulmonary embolism",          "ground_truth_icd11": "BB50.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "Hampton hump sign, Westermark sign"},
    {"case_id": "MIMIC-CXR-005", "image_path": "data/images/cxr_005.png",
     "ground_truth": "Pulmonary tuberculosis",      "ground_truth_icd11": "1B10.1",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "Bilateral upper zone infiltrates, early cavitation"},
    {"case_id": "MIMIC-CXR-006", "image_path": "data/images/cxr_006.png",
     "ground_truth": "Pulmonary sarcoidosis",       "ground_truth_icd11": "CB07.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "Bilateral hilar adenopathy, perilymphatic pattern"},
    {"case_id": "MIMIC-CXR-007", "image_path": "data/images/cxr_007.png",
     "ground_truth": "Pneumothorax",                "ground_truth_icd11": "CB01.0",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "",
     "notes": "Left sided pneumothorax, tracheal deviation"},
    {"case_id": "MIMIC-CXR-008", "image_path": "data/images/cxr_008.png",
     "ground_truth": "Pulmonary adenocarcinoma",   "ground_truth_icd11": "2C25.0",
     "organ": "lung",    "modality": "CT", "expected_risk_score": "Lung-RADS 4B",
     "notes": "14mm solid RUL nodule, SUV 4.2 on PET"},
    # ── MSK ───────────────────────────────────────────────────────────────────
    {"case_id": "RSNA-MSK-001",  "image_path": "data/images/msk_001.png",
     "ground_truth": "Rib fracture",                "ground_truth_icd11": "NB82.0",
     "organ": "msk",     "modality": "CXR", "expected_risk_score": "",
     "notes": "Right 5th rib cortical disruption"},
    {"case_id": "RSNA-MSK-002",  "image_path": "data/images/msk_002.png",
     "ground_truth": "Vertebral fracture",          "ground_truth_icd11": "NA83.0",
     "organ": "msk",     "modality": "CXR", "expected_risk_score": "",
     "notes": "L1 compression fracture Genant grade 2"},
    {"case_id": "RSNA-MSK-003",  "image_path": "data/images/msk_003.png",
     "ground_truth": "Hip fracture",                "ground_truth_icd11": "NB80.0",
     "organ": "msk",     "modality": "CXR", "expected_risk_score": "",
     "notes": "Right neck of femur fracture Garden III"},
    {"case_id": "RSNA-MSK-004",  "image_path": "data/images/msk_004.png",
     "ground_truth": "Osteoporosis",                "ground_truth_icd11": "FB83.1",
     "organ": "msk",     "modality": "CXR", "expected_risk_score": "",
     "notes": "Generalised reduced bone density, T-score -2.8"},
    # ── Thyroid / Breast / Liver ───────────────────────────────────────────────
    {"case_id": "TIRADS-001",    "image_path": "data/images/thy_001.png",
     "ground_truth": "Papillary thyroid carcinoma", "ground_truth_icd11": "2D10.0",
     "organ": "thyroid",  "modality": "US",  "expected_risk_score": "TR5",
     "notes": "Solid, hypoechoic, taller-than-wide, punctate calcifications"},
    {"case_id": "BIRADS-001",    "image_path": "data/images/brs_001.png",
     "ground_truth": "Invasive ductal carcinoma breast", "ground_truth_icd11": "2C61.0",
     "organ": "breast",  "modality": "MRI", "expected_risk_score": "5",
     "notes": "2.1cm spiculated mass, skin retraction"},
    {"case_id": "LIRADS-001",    "image_path": "data/images/liv_001.png",
     "ground_truth": "Hepatocellular carcinoma",    "ground_truth_icd11": "2C12.0",
     "organ": "liver",   "modality": "CT",  "expected_risk_score": "LR-5",
     "notes": "Arterial hyperenhancement + washout, 2.8cm cirrhotic liver"},
    # ── Neuro ─────────────────────────────────────────────────────────────────
    {"case_id": "NEURO-001",     "image_path": "data/images/neu_001.png",
     "ground_truth": "Glioblastoma",                "ground_truth_icd11": "2A00.0",
     "organ": "neuro",   "modality": "MRI", "expected_risk_score": "",
     "notes": "Ring-enhancing mass, central necrosis, butterfly pattern"},
    {"case_id": "NEURO-002",     "image_path": "data/images/neu_002.png",
     "ground_truth": "Ischaemic stroke",            "ground_truth_icd11": "8B20.0",
     "organ": "neuro",   "modality": "MRI", "expected_risk_score": "",
     "notes": "DWI restriction left MCA territory, 2h from onset"},
    {"case_id": "NEURO-003",     "image_path": "data/images/neu_003.png",
     "ground_truth": "Intracerebral haemorrhage",   "ground_truth_icd11": "8B00.0",
     "organ": "neuro",   "modality": "CT",  "expected_risk_score": "",
     "notes": "Right basal ganglia hyperdense lesion 35ml"},
    # ── Negative / Benign (to test false positive rate) ────────────────────────
    {"case_id": "NORMAL-001",    "image_path": "data/images/nrm_001.png",
     "ground_truth": "No significant abnormality",  "ground_truth_icd11": "N/A",
     "organ": "lung",    "modality": "CXR", "expected_risk_score": "Lung-RADS 1",
     "notes": "Normal CXR — negative control"},
    {"case_id": "NORMAL-002",    "image_path": "data/images/nrm_002.png",
     "ground_truth": "No significant abnormality",  "ground_truth_icd11": "N/A",
     "organ": "neuro",   "modality": "CT",  "expected_risk_score": "",
     "notes": "Normal CT head — negative control"},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset loader
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkDataset:
    """
    Load benchmark cases from a CSV file or the built-in synthetic dataset.

    CSV columns (header required):
        case_id, image_path, ground_truth, ground_truth_icd11,
        organ, modality, expected_risk_score, notes, prior_image_path
    """

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.cases: List[BenchmarkCase] = []
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)
        else:
            self._load_builtin()

    def _load_csv(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.cases.append(BenchmarkCase(
                    case_id=row.get("case_id", ""),
                    image_path=row.get("image_path", ""),
                    ground_truth=row.get("ground_truth", ""),
                    ground_truth_icd11=row.get("ground_truth_icd11", ""),
                    organ=row.get("organ", ""),
                    modality=row.get("modality", ""),
                    expected_risk_score=row.get("expected_risk_score", ""),
                    notes=row.get("notes", ""),
                    prior_image_path=row.get("prior_image_path", ""),
                ))

    def _load_builtin(self) -> None:
        for d in BUILTIN_CASES:
            self.cases.append(BenchmarkCase(**{
                k: d.get(k, "") for k in BenchmarkCase.__dataclass_fields__
            }))

    def __len__(self) -> int:
        return len(self.cases)

    def filter_organ(self, organ: str) -> "BenchmarkDataset":
        ds = BenchmarkDataset.__new__(BenchmarkDataset)
        ds.cases = [c for c in self.cases if c.organ.lower() == organ.lower()]
        return ds


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise(s: str) -> str:
    """Lower-case, strip, collapse whitespace for loose string matching."""
    return " ".join(s.lower().split())


def _top1_match(predicted: str, truth: str) -> bool:
    p, t = _normalise(predicted), _normalise(truth)
    return p == t or t in p or p in t


def _top3_match(ddx: List[Dict], truth: str) -> bool:
    t = _normalise(truth)
    for h in ddx[:3]:
        if t in _normalise(h.get("diagnosis", "")) or \
           _normalise(h.get("diagnosis", "")) in t:
            return True
    return False


def _compute_ece(results: List[CaseResult], n_bins: int = 10) -> float:
    """Expected Calibration Error over all cases."""
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    bw = 1.0 / n_bins
    for r in results:
        c = max(0.0, min(1.0 - 1e-9, r.sc))
        idx = min(n_bins - 1, int(c / bw))
        acc = 1.0 if r.top1_correct else 0.0
        bins[idx].append((c, acc))
    total = len(results)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if not b:
            continue
        mc = sum(x[0] for x in b) / len(b)
        ma = sum(x[1] for x in b) / len(b)
        ece += abs(ma - mc) * len(b) / total
    return ece


def _compute_brier(results: List[CaseResult]) -> float:
    """Brier score = mean((sc − correct)²)."""
    if not results:
        return 0.0
    return sum((r.sc - (1.0 if r.top1_correct else 0.0)) ** 2
               for r in results) / len(results)


def _compute_auc(results: List[CaseResult]) -> float:
    """
    Approximate macro one-vs-rest AUC using the trapezoidal rule.
    We treat each case as binary (correct = positive) sorted by Sc.
    """
    if not results:
        return 0.0
    pairs = sorted(((r.sc, r.top1_correct) for r in results), key=lambda x: -x[0])
    tp, fp, fn = 0, 0, sum(1 for r in results if r.top1_correct)
    tn = len(results) - fn
    tpr_pts, fpr_pts = [0.0], [0.0]
    for _, correct in pairs:
        if correct:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_pts.append(tpr)
        fpr_pts.append(fpr)
    tpr_pts.append(1.0); fpr_pts.append(1.0)
    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_pts)):
        auc += (fpr_pts[i] - fpr_pts[i-1]) * (tpr_pts[i] + tpr_pts[i-1]) / 2.0
    return max(0.0, min(1.0, abs(auc)))


def _compute_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = (2 * precision * recall / max(1e-9, precision + recall))
    return precision, recall, f1


def _aggregate(results: List[CaseResult]) -> BenchmarkMetrics:
    """Compute all aggregate metrics from a list of CaseResult."""
    n = len(results)
    if n == 0:
        return BenchmarkMetrics()

    top1  = sum(1 for r in results if r.top1_correct)
    top3  = sum(1 for r in results if r.top3_correct)
    icd11 = sum(1 for r in results if r.icd11_correct)
    escs  = sum(1 for r in results if r.escalated)

    tp = sum(1 for r in results if r.top1_correct and not r.escalated)
    fp = sum(1 for r in results if not r.top1_correct and not r.escalated)
    fn = sum(1 for r in results if r.top1_correct and r.escalated)

    prec, rec, f1 = _compute_f1(tp, fp, fn)
    # Type-I  = FPR = FP / (FP + TN)
    # Type-II = FNR = FN / (TP + FN)
    tn = n - tp - fp - fn
    type1 = fp / max(1, fp + tn)
    type2 = fn / max(1, tp + fn)

    return BenchmarkMetrics(
        n_cases=n,
        accuracy=top1 / n,
        top3_accuracy=top3 / n,
        precision=prec,
        recall=rec,
        f1=f1,
        auc=_compute_auc(results),
        ece=_compute_ece(results),
        brier=_compute_brier(results),
        type1_error=type1,
        type2_error=type2,
        escalation_rate=escs / n,
        mean_iterations=sum(r.iterations for r in results) / n,
        mean_sc=sum(r.sc for r in results) / n,
        mean_sigma2=sum(r.sigma2 for r in results) / n,
        mean_elapsed_s=sum(r.elapsed_s for r in results) / n,
        icd11_accuracy=icd11 / n,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """
    Run the full R-MoE pipeline on every case in a BenchmarkDataset and
    compute the paper's reported evaluation metrics.

    Usage:
        from rmoe.eval import BenchmarkRunner, BenchmarkDataset
        runner  = BenchmarkRunner(mr_tom)
        dataset = BenchmarkDataset("data/benchmark_cases.csv")
        results = runner.run(dataset)
        runner.print_report(results)
        runner.save_results(results, "benchmark_results.json")
    """

    def __init__(self, mr_tom, verbose: bool = True) -> None:
        """
        Args:
            mr_tom:  An initialised MrTom instance (from rmoe.core).
            verbose: Print per-case progress.
        """
        self._mr_tom  = mr_tom
        self._verbose = verbose

    def run(
        self,
        dataset: BenchmarkDataset,
        max_cases: Optional[int] = None,
    ) -> List[CaseResult]:
        """
        Run the pipeline on every case and return per-case CaseResult list.

        Args:
            dataset:    BenchmarkDataset to evaluate.
            max_cases:  Optional limit (useful for quick smoke-tests).
        """
        from rmoe.ui import CYAN, GREEN, RED, YELLOW, BOLD, DIM, RESET, _rule
        from rmoe.ontology import lookup_icd11

        cases = dataset.cases
        if max_cases:
            cases = cases[:max_cases]

        results: List[CaseResult] = []
        total   = len(cases)

        print(f"\n{BOLD}  Running Benchmark  ({total} cases){RESET}")
        _rule()
        print(
            f"  {'#':>4}  {'Case ID':<20} {'Organ':<8} {'Mod':<4} "
            f"{'Sc':>6}  {'Itr':>3}  {'T1':>4}  {'T3':>4}  {'t(s)':>5}"
        )
        _rule()

        for i, case in enumerate(cases, 1):
            t0 = time.time()
            result = CaseResult(case=case)

            try:
                summary = self._mr_tom.process_patient_case(
                    image_path=case.image_path,
                    prior_image=case.prior_image_path or None,
                    audit_log_path=None,
                )

                # Pull data from summary
                result.iterations = summary.iterations_executed
                result.elapsed_s  = time.time() - t0
                result.escalated  = summary.escalated_to_human

                if summary.trace:
                    last = summary.trace[-1]
                    result.sc     = last.metrics.confidence
                    result.sigma2 = last.metrics.ddx_variance
                    ddx_list = last.ddx_ensemble.get("hypotheses", [])
                    result.all_ddx = ddx_list
                    if ddx_list:
                        top = max(ddx_list, key=lambda h: h["probability"])
                        result.predicted_diagnosis  = top["diagnosis"]
                        result.predicted_probability = top["probability"]

                # ICD-11 from final report
                if summary.final_report_json:
                    try:
                        rep = json.loads(summary.final_report_json)
                        result.icd11_predicted  = rep.get("standard", "")
                        rs = rep.get("risk_stratification", {})
                        result.risk_score_predicted = (
                            rs.get("score", "") if isinstance(rs, dict) else ""
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass

            except Exception as exc:
                result.error    = str(exc)
                result.elapsed_s = time.time() - t0

            # Derive match flags
            result.top1_correct = _top1_match(
                result.predicted_diagnosis, case.ground_truth
            )
            result.top3_correct = _top3_match(
                result.all_ddx, case.ground_truth
            )
            result.icd11_correct = (
                case.ground_truth_icd11 != "N/A" and
                case.ground_truth_icd11 in result.icd11_predicted
            )

            results.append(result)

            if self._verbose:
                t1c = GREEN if result.top1_correct else RED
                t3c = GREEN if result.top3_correct else YELLOW
                sc_c = GREEN if result.sc >= 0.90 else YELLOW
                err  = f"  {RED}ERR{RESET}" if result.error else ""
                print(
                    f"  {i:>4}  {case.case_id:<20} {case.organ:<8} {case.modality:<4} "
                    f"{sc_c}{result.sc:>6.4f}{RESET}  {result.iterations:>3}  "
                    f"{t1c}{'✓' if result.top1_correct else '✗':>4}{RESET}  "
                    f"{t3c}{'✓' if result.top3_correct else '✗':>4}{RESET}  "
                    f"{result.elapsed_s:>5.1f}{err}"
                )

        _rule()
        return results

    # ── Report ────────────────────────────────────────────────────────────────

    def print_report(self, results: List[CaseResult]) -> None:
        from rmoe.ui import (CYAN, GREEN, RED, YELLOW, BOLD, DIM, RESET,
                              WHITE, MAGENTA, _rule, _kv)

        metrics = _aggregate(results)
        print(f"\n{BOLD}  ╔══ BENCHMARK RESULTS ═══════════════════════════════════════╗{RESET}")
        _rule()

        # Main metrics
        fc = lambda v, thr: GREEN if v >= thr else (YELLOW if v >= thr * 0.9 else RED)
        print(f"\n  {BOLD}Classification Metrics:{RESET}")
        _kv("Cases evaluated",    str(metrics.n_cases))
        _kv("Top-1 Accuracy",     f"{metrics.accuracy:.4f}  ({metrics.accuracy*100:.1f}%)",
            vc=fc(metrics.accuracy, 0.80))
        _kv("Top-3 Accuracy",     f"{metrics.top3_accuracy:.4f}  ({metrics.top3_accuracy*100:.1f}%)",
            vc=fc(metrics.top3_accuracy, 0.90))
        _kv("Precision",          f"{metrics.precision:.4f}", vc=fc(metrics.precision, 0.80))
        _kv("Recall",             f"{metrics.recall:.4f}",    vc=fc(metrics.recall, 0.80))
        _kv("F1-Score",           f"{metrics.f1:.4f}  (paper target: 0.92)",
            vc=fc(metrics.f1, 0.90))
        _kv("AUC (macro OvR)",    f"{metrics.auc:.4f}",       vc=fc(metrics.auc, 0.85))

        print(f"\n  {BOLD}Calibration Metrics:{RESET}")
        ece_c = GREEN if metrics.ece <= 0.10 else (YELLOW if metrics.ece <= 0.15 else RED)
        _kv("ECE",                f"{metrics.ece:.4f}  (paper target: ≤ 0.08)", vc=ece_c)
        _kv("Brier Score",        f"{metrics.brier:.4f}  (0=perfect)",
            vc=GREEN if metrics.brier <= 0.15 else YELLOW)
        _kv("ICD-11 Accuracy",    f"{metrics.icd11_accuracy:.4f}", vc=fc(metrics.icd11_accuracy, 0.70))

        print(f"\n  {BOLD}Safety / Error Rates:{RESET}")
        t1c = GREEN if metrics.type1_error <= 0.08 else (YELLOW if metrics.type1_error <= 0.12 else RED)
        t2c = GREEN if metrics.type2_error <= 0.10 else YELLOW
        _kv("Type-I  (FPR)",      f"{metrics.type1_error:.4f}  = {metrics.type1_error*100:.1f}%  (paper target: ≤ 5.2%)", vc=t1c)
        _kv("Type-II (FNR)",      f"{metrics.type2_error:.4f}  = {metrics.type2_error*100:.1f}%", vc=t2c)
        _kv("Escalation Rate",    f"{metrics.escalation_rate:.4f}  = {metrics.escalation_rate*100:.1f}%",
            vc=YELLOW if metrics.escalation_rate > 0.20 else GREEN)

        print(f"\n  {BOLD}Pipeline Behaviour:{RESET}")
        _kv("Mean Iterations",    f"{metrics.mean_iterations:.2f} / 3",   vc=CYAN)
        _kv("Mean Sc",            f"{metrics.mean_sc:.4f}",               vc=CYAN)
        _kv("Mean σ²",            f"{metrics.mean_sigma2:.6f}",           vc=DIM)
        _kv("Mean Inference (s)", f"{metrics.mean_elapsed_s:.1f} s",      vc=DIM)

        print()
        _rule()
        self._print_comparison_table(metrics)
        print()
        _rule()

    def _print_comparison_table(self, m: BenchmarkMetrics) -> None:
        from rmoe.ui import GREEN, BOLD, DIM, RESET, CYAN, YELLOW, RED

        # Paper baselines (Table 1)
        _GPT4V  = dict(f1=0.85, type1=0.078, ece=0.15, t=32.0)
        _GEMINI = dict(f1=0.87, type1=0.071, ece=0.13, t=38.0)

        print(f"\n  {BOLD}Comparison vs Paper Baselines (Table 1){RESET}")
        print(
            f"  {'Metric':<24} {'R-MoE (this run)':>18} {'R-MoE (paper)':>16}"
            f" {'GPT-4V':>10} {'Gemini 1.5':>12}"
        )
        print("  " + "─" * 84)

        rows = [
            ("F1-Score",       m.f1,             0.92, 0.85, 0.87, True),
            ("Type-I Error %", m.type1_error*100, 5.2,  7.8,  7.1,  False),
            ("ECE",            m.ece,             0.08, 0.15, 0.13, False),
            ("Mean Iter",      m.mean_iterations, None, None, None, None),
            ("Escalation %",   m.escalation_rate*100, None, None, None, None),
        ]
        for name, val, paper, gpt4v, gemini, higher_better in rows:
            def fc(v, ref, hb):
                if ref is None:
                    return CYAN
                return (GREEN + BOLD if (v >= ref if hb else v <= ref) else DIM)

            gpt_str    = f"{gpt4v:>10.2f}"   if gpt4v  is not None else f"{'—':>10}"
            gem_str    = f"{gemini:>12.2f}"   if gemini is not None else f"{'—':>12}"
            paper_str  = f"{paper:>16.2f}"    if paper  is not None else f"{'—':>16}"
            run_c      = fc(val, paper, higher_better)

            print(
                f"  {name:<24} {run_c}{val:>18.4f}{RESET}"
                f" {paper_str} {gpt_str} {gem_str}"
            )

    def print_latex(self, results: List[CaseResult]) -> str:
        """Return a LaTeX tabular block (paper Table 1 row for R-MoE)."""
        m = _aggregate(results)
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \begin{tabular}{lcccccc}",
            r"    \hline",
            r"    \textbf{System} & \textbf{F1} & \textbf{Prec} & \textbf{Recall}"
            r" & \textbf{AUC} & \textbf{ECE} & \textbf{TypeI \%} \\",
            r"    \hline",
            f"    R-MoE (this run) & {m.f1:.3f} & {m.precision:.3f}"
            f" & {m.recall:.3f} & {m.auc:.3f} & {m.ece:.3f}"
            f" & {m.type1_error*100:.1f} \\\\",
            r"    R-MoE (paper)    & 0.920 & --- & --- & --- & 0.080 & 5.2 \\",
            r"    GPT-4V           & 0.850 & --- & --- & --- & 0.150 & 7.8 \\",
            r"    Gemini 1.5 Pro   & 0.870 & --- & --- & --- & 0.130 & 7.1 \\",
            r"    \hline",
            r"  \end{tabular}",
            r"  \caption{R-MoE benchmark vs baselines (MIMIC-CXR + RSNA)}",
            r"  \label{tab:benchmark}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def save_results(
        self, results: List[CaseResult], path: str
    ) -> None:
        """Save full results + metrics to a JSON file."""
        m = _aggregate(results)
        out = {
            "metrics": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in m.__dict__.items()
            },
            "cases": [
                {
                    "case_id":               r.case.case_id,
                    "ground_truth":          r.case.ground_truth,
                    "predicted":             r.predicted_diagnosis,
                    "predicted_probability": round(r.predicted_probability, 4),
                    "sc":                    round(r.sc, 4),
                    "sigma2":                round(r.sigma2, 6),
                    "iterations":            r.iterations,
                    "elapsed_s":             round(r.elapsed_s, 2),
                    "top1_correct":          r.top1_correct,
                    "top3_correct":          r.top3_correct,
                    "icd11_correct":         r.icd11_correct,
                    "escalated":             r.escalated,
                    "icd11_predicted":       r.icd11_predicted,
                    "risk_score_predicted":  r.risk_score_predicted,
                    "error":                 r.error,
                }
                for r in results
            ],
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
        except OSError as exc:
            import sys
            print(f"[eval] save failed: {exc}", file=sys.stderr)
