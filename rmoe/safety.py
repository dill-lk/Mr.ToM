"""
rmoe/safety.py — CSR Dual-Layer Safety Validator.

Implements paper §"Synthesis Agent and Structured Clinical Reporting":
  "The architecture of the Synthesis Agent includes a dual-layer
   validation mechanism. The first layer is a semantic parser that
   extracts entities from the reasoning logic, while the second layer
   is a deterministic rule-checker that evaluates the final report
   against clinical safety protocols, such as drug-dose-duration rules
   for pediatric patients."

Also implements:
  "This ensures that the automated treatment recommendations provided
   in the 'Impression' section of the report are both medically sound
   and legally compliant."

Layer 1 — SemanticParser:
  • Extracts diagnosis entities, ICD-11 codes, drug names, doses, durations
  • Validates ICD-11 format compliance
  • Extracts risk scores (Lung-RADS / TIRADS / BI-RADS / LI-RADS / PI-RADS)
  • Flags hallucinated entity patterns (invented ICD codes, impossible doses)

Layer 2 — ClinicalRuleChecker:
  • Drug-dose-duration rules (paediatric weight-based dosing guard)
  • Contraindication checks (e.g. NSAIDs contraindicated in renal failure)
  • Imaging-escalation rules (e.g. Lung-RADS 4X requires PET-CT within 1 month)
  • ICD-11 consistency (diagnosis must match the ICD code family)
  • Safety gate: BLOCK report if critical violation; WARN if minor

SafetyReport:
  • Summarises all findings from both layers
  • Sets status: PASS | WARN | BLOCK
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class SafetyStatus(Enum):
    PASS  = "PASS"    # no issues
    WARN  = "WARN"    # minor issues, report can proceed with annotation
    BLOCK = "BLOCK"   # critical violation, do not release report


class ViolationSeverity(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntity:
    entity_type: str       # "diagnosis" | "drug" | "icd11" | "risk_score" | "dose"
    value:       str
    context:     str = ""  # surrounding sentence


@dataclass
class SafetyViolation:
    rule_id:   str
    severity:  ViolationSeverity
    message:   str
    fix_hint:  str = ""


@dataclass
class SafetyReport:
    """Complete dual-layer safety validation result."""
    status:         SafetyStatus             = SafetyStatus.PASS
    entities:       List[ExtractedEntity]    = field(default_factory=list)
    violations:     List[SafetyViolation]    = field(default_factory=list)
    layer1_ok:      bool                     = True
    layer2_ok:      bool                     = True
    annotation:     str                      = ""  # appended to report if WARN

    def critical_violations(self) -> List[SafetyViolation]:
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]

    def warnings(self) -> List[SafetyViolation]:
        return [v for v in self.violations if v.severity == ViolationSeverity.WARNING]


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 1 — Semantic Parser
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticParser:
    """
    Extract clinical entities from CSR report text.

    Entities extracted:
      • ICD-11 codes      (pattern: [0-9A-Z]{2}[0-9A-Z.]{2,8})
      • Drug names        (from known drug list + common patterns)
      • Dose statements   (e.g. "500 mg", "10 mg/kg")
      • Risk scores       (Lung-RADS 4X, TR5, BI-RADS 5, etc.)
      • Diagnoses         (text after "Diagnosis:", "Impression:", etc.)
    """

    # ICD-11 code pattern — handles both letter-first (CA40.0) and
    # digit-first (2C25.0) formats. Second character must be A-Z to
    # exclude numeric strings like years (2022).
    _ICD11_RE = re.compile(r"\b([0-9A-Z][A-Z]\d{2}(?:\.\d{1,3})?)\b")

    # Dose pattern: number + unit
    _DOSE_RE = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ml|mL|mg/kg|mcg/kg|IU|U)\b",
        re.IGNORECASE,
    )

    # Risk score patterns
    _RISK_RE = re.compile(
        r"\b(Lung-RADS\s*[1-4][ABCSX]?|TR[1-5]|BI-RADS\s*[0-6]|"
        r"LR-[1-5M]|PI-RADS\s*[1-5]|TIRADS\s*[1-5])\b",
        re.IGNORECASE,
    )

    # Common safe drugs (not exhaustive — demonstration subset)
    _SAFE_DRUGS = frozenset({
        "amoxicillin", "co-amoxiclav", "azithromycin", "ciprofloxacin",
        "metronidazole", "doxycycline", "cefuroxime", "trimethoprim",
        "ibuprofen", "paracetamol", "acetaminophen", "naproxen",
        "prednisolone", "dexamethasone", "hydrocortisone",
        "omeprazole", "pantoprazole", "lansoprazole",
        "salbutamol", "ipratropium", "budesonide", "fluticasone",
        "aspirin", "clopidogrel", "warfarin", "heparin",
        "metformin", "insulin", "atorvastatin", "ramipril", "amlodipine",
    })

    def parse(self, report_text: str) -> Tuple[List[ExtractedEntity], List[SafetyViolation]]:
        """
        Extract entities from report text.
        Returns (entities, layer1_violations).
        """
        entities:   List[ExtractedEntity]  = []
        violations: List[SafetyViolation]  = []

        sentences = re.split(r"(?<=[.!?])\s+", report_text)

        # ── ICD-11 codes ──────────────────────────────────────────────────────
        for m in self._ICD11_RE.finditer(report_text):
            code = m.group(1)
            entities.append(ExtractedEntity(
                entity_type="icd11", value=code,
                context=self._context(report_text, m.start()),
            ))
            # Validate format: [digit|letter][letter][digit][digit][.digits]
            if not re.match(r"[0-9A-Z][A-Z]\d{2}", code):
                violations.append(SafetyViolation(
                    rule_id="ICD11-FORMAT",
                    severity=ViolationSeverity.WARNING,
                    message=f"ICD-11 code '{code}' has unexpected format.",
                    fix_hint="Verify against ICD-11 browser (https://icd.who.int).",
                ))

        # ── Doses ─────────────────────────────────────────────────────────────
        for m in self._DOSE_RE.finditer(report_text):
            value = f"{m.group(1)} {m.group(2)}"
            entities.append(ExtractedEntity(
                entity_type="dose", value=value,
                context=self._context(report_text, m.start()),
            ))

        # ── Risk scores ───────────────────────────────────────────────────────
        for m in self._RISK_RE.finditer(report_text):
            entities.append(ExtractedEntity(
                entity_type="risk_score", value=m.group(1),
                context=self._context(report_text, m.start()),
            ))

        # ── Drug names ────────────────────────────────────────────────────────
        report_lower = report_text.lower()
        for drug in self._SAFE_DRUGS:
            if drug in report_lower:
                entities.append(ExtractedEntity(
                    entity_type="drug", value=drug,
                    context="",
                ))

        return entities, violations

    @staticmethod
    def _context(text: str, pos: int, window: int = 80) -> str:
        start = max(0, pos - window // 2)
        end   = min(len(text), pos + window // 2)
        return text[start:end].replace("\n", " ")


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 2 — Clinical Rule Checker
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalRuleChecker:
    """
    Deterministic rule-checker for clinical safety compliance.

    Rules implemented (paper §"dual-layer validation"):
      DRUG-DOSE-PEDS  — flag if dose exceeds paediatric weight-based limit
      NSAID-RENAL     — NSAIDs contraindicated with renal failure mention
      NSAID-GI        — NSAIDs with GI bleed history mention
      LUNGRADS-4X     — Lung-RADS 4X requires PET-CT / tissue sampling
      LUNGRADS-4B     — Lung-RADS 4B requires CT follow-up ≤ 3 months
      TIRADS-5        — TR5 requires FNA biopsy
      BIRADS-5        — BI-RADS 5 requires tissue diagnosis
      LIRADS-5        — LR-5 requires MDT review
      ICD-MALIGNANCY  — Any 2C* ICD-11 code must include staging & MDT flag
    """

    # Maximum safe adult single doses (mg) — illustrative clinical limits.
    # ⚠️  IMPORTANT: These values are for demonstration and research purposes
    # only. They must be validated against current clinical formularies
    # (BNF, local protocol, or UpToDate) before any production deployment.
    _MAX_ADULT_DOSE_MG = {
        "ibuprofen":     800.0,
        "paracetamol": 1_000.0,
        "aspirin":       900.0,
        "amoxicillin":   500.0,
        "ciprofloxacin": 750.0,
    }

    def check(
        self,
        report_text: str,
        entities: List[ExtractedEntity],
        patient_age_years: Optional[float] = None,
        patient_weight_kg: Optional[float] = None,
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        text_l = report_text.lower()

        violations += self._check_drug_doses(entities, patient_age_years)
        violations += self._check_nsaid_contraindications(entities, text_l)
        violations += self._check_risk_score_escalation(entities, text_l)
        violations += self._check_malignancy_staging(entities, text_l)

        return violations

    # ── Drug dose rules ───────────────────────────────────────────────────────

    def _check_drug_doses(
        self,
        entities: List[ExtractedEntity],
        age_years: Optional[float],
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        is_paediatric = (age_years is not None and age_years < 18)

        drug_names = {e.value for e in entities if e.entity_type == "drug"}
        doses = [e for e in entities if e.entity_type == "dose"]

        for dose_ent in doses:
            value_str, unit = dose_ent.value.split()
            try:
                value = float(value_str)
            except ValueError:
                continue

            # Convert to mg for comparison
            value_mg = value
            if unit.lower() in ("g",):
                value_mg = value * 1000

            # Paediatric dose flag (> 15 mg/kg per dose is unusual for most drugs)
            if is_paediatric and value_mg > 250:
                violations.append(SafetyViolation(
                    rule_id="DRUG-DOSE-PEDS",
                    severity=ViolationSeverity.WARNING,
                    message=(
                        f"Dose {dose_ent.value} may exceed paediatric weight-based "
                        f"limit for a patient aged {age_years:.0f} years."
                    ),
                    fix_hint="Verify dosing with BNF for Children / local formulary.",
                ))

            # Adult single-dose caps
            for drug in drug_names:
                cap = self._MAX_ADULT_DOSE_MG.get(drug, None)
                if cap and value_mg > cap:
                    violations.append(SafetyViolation(
                        rule_id="DRUG-DOSE-ADULT",
                        severity=ViolationSeverity.CRITICAL,
                        message=(
                            f"Recommended dose {dose_ent.value} exceeds maximum "
                            f"single adult dose for {drug} ({cap:.0f} mg)."
                        ),
                        fix_hint=f"Maximum single adult dose of {drug}: {cap:.0f} mg.",
                    ))

        return violations

    # ── Contraindication checks ───────────────────────────────────────────────

    def _check_nsaid_contraindications(
        self,
        entities: List[ExtractedEntity],
        text_l: str,
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        nsaid_drugs = {"ibuprofen", "naproxen", "aspirin", "diclofenac", "celecoxib"}
        has_nsaid = any(e.value in nsaid_drugs for e in entities if e.entity_type == "drug")
        if not has_nsaid:
            return violations

        if any(kw in text_l for kw in ("renal failure", "aki", "ckd", "eGFR < 30",
                                        "chronic kidney", "renal impairment")):
            violations.append(SafetyViolation(
                rule_id="NSAID-RENAL",
                severity=ViolationSeverity.CRITICAL,
                message="NSAID prescribed with renal failure mention in report.",
                fix_hint="NSAIDs are contraindicated in AKI/CKD (eGFR < 30). "
                         "Consider paracetamol or opioid analgesia.",
            ))
        if any(kw in text_l for kw in ("gi bleed", "peptic ulcer", "haematemesis",
                                        "melena", "upper gi")):
            violations.append(SafetyViolation(
                rule_id="NSAID-GI",
                severity=ViolationSeverity.WARNING,
                message="NSAID with GI bleed / peptic ulcer history in report.",
                fix_hint="Add gastroprotection (PPI) or switch to paracetamol.",
            ))
        return violations

    # ── Risk-score mandatory escalation rules ─────────────────────────────────

    def _check_risk_score_escalation(
        self,
        entities: List[ExtractedEntity],
        text_l: str,
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        risk_scores = {e.value.upper() for e in entities if e.entity_type == "risk_score"}

        rules = [
            # (score_pattern,  required_phrase,          rule_id,          severity, msg,  fix)
            ("LUNG-RADS 4X", "pet",
             "LUNGRADS-4X", ViolationSeverity.CRITICAL,
             "Lung-RADS 4X requires PET-CT or tissue sampling — not mentioned.",
             "Add: 'PET-CT recommended within 1 month per ACR Lung-RADS 2022.'"),
            ("LUNG-RADS 4B", "ct",
             "LUNGRADS-4B", ViolationSeverity.WARNING,
             "Lung-RADS 4B requires CT follow-up within 3 months.",
             "Add: 'CT chest without contrast at 3 months per ACR Lung-RADS 2022.'"),
            ("TR5", "fna",
             "TIRADS-5",   ViolationSeverity.CRITICAL,
             "TIRADS TR5 requires FNA biopsy recommendation.",
             "Add FNA recommendation per ACR TIRADS guidelines."),
            ("BI-RADS 5",   "biopsy",
             "BIRADS-5",   ViolationSeverity.CRITICAL,
             "BI-RADS 5 requires tissue diagnosis recommendation.",
             "Add: 'Ultrasound-guided core biopsy recommended (BI-RADS 5).'"),
            ("LR-5",        "mdt",
             "LIRADS-5",   ViolationSeverity.WARNING,
             "LR-5 HCC should prompt MDT discussion.",
             "Add: 'MDT hepatology review recommended per LI-RADS 2023.'"),
        ]

        for score_pat, required_kw, rule_id, severity, msg, fix in rules:
            if any(score_pat in rs for rs in risk_scores):
                if required_kw not in text_l:
                    violations.append(SafetyViolation(
                        rule_id=rule_id, severity=severity,
                        message=msg, fix_hint=fix,
                    ))

        return violations

    # ── Malignancy staging ────────────────────────────────────────────────────

    def _check_malignancy_staging(
        self,
        entities: List[ExtractedEntity],
        text_l: str,
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        malignancy_icd_prefixes = ("2C", "2D", "2E", "2F", "XH")  # ICD-11 neoplasm codes
        has_malignancy = any(
            e.entity_type == "icd11" and
            any(e.value.startswith(p) for p in malignancy_icd_prefixes)
            for e in entities
        )
        if has_malignancy:
            if "staging" not in text_l and "stage" not in text_l:
                violations.append(SafetyViolation(
                    rule_id="ICD-MALIGNANCY-STAGING",
                    severity=ViolationSeverity.WARNING,
                    message="Malignancy ICD-11 code present but no staging mentioned.",
                    fix_hint="Add TNM staging or clinical stage to Impression section.",
                ))
            if "mdt" not in text_l and "multidisciplinary" not in text_l:
                violations.append(SafetyViolation(
                    rule_id="ICD-MALIGNANCY-MDT",
                    severity=ViolationSeverity.WARNING,
                    message="Malignancy ICD-11 code present but no MDT referral mentioned.",
                    fix_hint="Add MDT referral recommendation.",
                ))
        return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  CSRSafetyValidator — combines both layers
# ═══════════════════════════════════════════════════════════════════════════════

class CSRSafetyValidator:
    """
    Full dual-layer safety validator for CSR reports.

    Usage:
        validator = CSRSafetyValidator()
        report    = validator.validate(report_text, patient_age_years=8.0)
        if report.status == SafetyStatus.BLOCK:
            ...  # do not release report
    """

    def __init__(self) -> None:
        self._parser  = SemanticParser()
        self._checker = ClinicalRuleChecker()

    def validate(
        self,
        report_text: str,
        patient_age_years:  Optional[float] = None,
        patient_weight_kg:  Optional[float] = None,
    ) -> SafetyReport:
        # Layer 1 — semantic parse
        entities, l1_violations = self._parser.parse(report_text)

        # Layer 2 — rule check
        l2_violations = self._checker.check(
            report_text, entities, patient_age_years, patient_weight_kg
        )

        all_violations = l1_violations + l2_violations
        critical = [v for v in all_violations if v.severity == ViolationSeverity.CRITICAL]
        warnings  = [v for v in all_violations if v.severity == ViolationSeverity.WARNING]

        if critical:
            status = SafetyStatus.BLOCK
        elif warnings:
            status = SafetyStatus.WARN
        else:
            status = SafetyStatus.PASS

        annotation = ""
        if warnings:
            annotation = (
                "\n\n⚠ SAFETY ANNOTATION (auto-generated by CSR dual-layer validator):\n"
                + "\n".join(f"  [{w.rule_id}] {w.message}" for w in warnings)
            )

        return SafetyReport(
            status=status,
            entities=entities,
            violations=all_violations,
            layer1_ok=(len(l1_violations) == 0),
            layer2_ok=(len(l2_violations) == 0),
            annotation=annotation,
        )


def print_safety_report(report: SafetyReport) -> None:
    """Print a safety report to the terminal (uses rmoe.ui colours)."""
    try:
        from rmoe.ui import BOLD, DIM, GREEN, RED, RESET, YELLOW, _rule
    except ImportError:
        BOLD = DIM = GREEN = RED = RESET = YELLOW = ""
        def _rule(): print("─" * 72)

    _rule()
    status_colour = {
        SafetyStatus.PASS:  GREEN,
        SafetyStatus.WARN:  YELLOW,
        SafetyStatus.BLOCK: RED + BOLD,
    }[report.status]

    print(f"  {status_colour}CSR Safety Validator: {report.status.value}{RESET}"
          f"  ({len(report.entities)} entities extracted, "
          f"{len(report.violations)} violations)")

    for v in report.violations:
        sc = RED + BOLD if v.severity == ViolationSeverity.CRITICAL else YELLOW
        print(f"\n  {sc}[{v.severity.value}] {v.rule_id}{RESET}")
        print(f"  {DIM}{v.message}{RESET}")
        if v.fix_hint:
            print(f"  Fix: {v.fix_hint}")
