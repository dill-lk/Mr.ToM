#include "ExpertInterface.hpp"
#include "CliOutput.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace {
void print_usage() {
    std::cout
        << "Usage:\n"
        << "  ./rmoe_engine --model-vision <path> --model-proj <path>"
           " --model-reasoning <path> --model-clinical <path> --image <path> [options]\n"
        << "\n"
        << "  Model paths (long-form aliases also accepted):\n"
        << "    --model-vision   / --vision-text   : vision LLM   (vision_text.gguf)\n"
        << "    --model-proj     / --vision-proj   : CLIP mmproj  (vision_proj.gguf)\n"
        << "    --model-reasoning / --reasoning    : ARLL model   (reasoning_expert.gguf)\n"
        << "    --model-clinical / --clinical      : CSR model    (clinical_expert.gguf)\n"
        << "\n"
        << "  Other options:\n"
        << "    --image   <path>  : patient image file (required)\n"
        << "    --settings <json> : JSON settings file\n"
        << "    --temp    <f>     : sampling temperature (default 0.2)\n"
        << "    --n-predict <n>   : max tokens to generate (default 128)\n"
        << "    --chat-target reasoning|clinical\n";
}

// Display the final clinical report by parsing the JSON and printing key fields.
void print_clinical_report(const std::string& report_json) {
    using namespace rmoe::cli;
    std::cout << '\n';
    print_section("CLINICAL REPORT");
    std::cout << '\n';
    try {
        const nlohmann::json rep = nlohmann::json::parse(report_json);

        print_kv("Standard",   rep.value("standard",  "N/A"), kCyan);
        print_kv("SNOMED CT",  rep.value("snomed_ct", "N/A"), kCyan);

        if (rep.contains("risk_stratification")) {
            const auto& rs = rep["risk_stratification"];
            print_kv("TIRADS",   rs.value("tirads", "N/A"), kYellow);
            print_kv("BI-RADS",  rs.value("birads", "N/A"), kYellow);
        }

        // Truncate narrative to keep terminal readable
        std::string narr = rep.value("narrative", "N/A");
        if (narr.size() > rmoe::cli::kMaxNarrativeLen) {
            narr = narr.substr(0U, rmoe::cli::kMaxNarrativeLen) + " ...";
        }
        print_kv("Narrative",  narr);
        print_kv("Treatment",  rep.value("treatment_recommendations", "N/A"));
        print_kv("Summary",    rep.value("summary", "N/A"));

        const bool hitl = rep.value("hitl_review_required", false);
        print_kv("HITL Review", hitl ? "Required" : "Not required",
                 hitl ? kRed : kGreen);
        if (hitl && rep.contains("hitl_reason")) {
            print_kv("HITL Reason", rep["hitl_reason"].get<std::string>(), kRed);
        }
    } catch (...) {
        // Fallback: raw JSON (shouldn't happen in practice)
        std::cout << kDim << report_json << rmoe::cli::kReset << '\n';
    }
    std::cout << '\n';
    print_rule('=');
}

// Display the run summary + iteration trace table.
void print_run_summary(const rmoe::RunSummary& summary, int max_iter) {
    using namespace rmoe::cli;
    std::cout << '\n';
    print_section("DIAGNOSTIC RUN SUMMARY");
    std::cout << '\n';

    const char* status_color = summary.success ? kGreen : kRed;
    const char* status_text  = summary.success
        ? "SUCCESS"
        : (summary.escalated_to_human ? "ESCALATED TO HUMAN" : "FAILED");

    print_kv("Result",     status_text,  status_color);
    print_kv("Escalated",  summary.escalated_to_human ? "Yes" : "No",
             summary.escalated_to_human ? kYellow : kGreen);

    std::ostringstream iters;
    iters << summary.iterations_executed << " / " << max_iter;
    print_kv("Iterations", iters.str(), kCyan);

    if (!summary.trace.empty()) {
        std::cout << '\n'
                  << kDim << "  Iteration Trace\n" << kReset;
        print_dash_rule();
        std::cout
                  << kBold << std::left
                  << "   #   " << std::setw(26) << "Decision"
                  << std::right
                  << std::setw(10) << "Sc"
                  << std::setw(10) << "sigma^2"
                  << std::setw(10) << "H"
                  << kReset << '\n';
        print_dash_rule();

        for (const auto& t : summary.trace) {
            const char* row_color = (t.metrics.confidence >= 0.90F) ? kGreen : kYellow;
            std::cout << row_color << std::fixed << std::setprecision(4)
                      << "   " << t.iteration << "   "
                      << std::left << std::setw(26) << t.decision
                      << std::right
                      << std::setw(10) << t.metrics.confidence
                      << std::setw(10) << t.metrics.ddx_variance
                      << std::setw(10) << t.metrics.predictive_entropy
                      << kReset << '\n';
        }

        print_dash_rule();
    }
}
} // namespace

int main(int argc, char** argv) {
    constexpr int kHardLimitIterations = 3;
    constexpr rmoe::ConfidenceScore kThreshold = 0.90F;

    std::string vision_proj;
    std::string vision_text;
    std::string reasoning;
    std::string clinical;
    std::string image_path;
    std::string settings_path;
    float       cli_temperature  = -1.0F; // <0 means "not set by CLI"
    int         cli_max_tokens   = -1;    // <0 means "not set by CLI"
    rmoe::ExpertTarget chat_target = rmoe::ExpertTarget::Reasoning;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        // vision projection / mmproj (CLIP)
        if ((arg == "--vision-proj" || arg == "--model-proj") && i + 1 < argc) {
            vision_proj = argv[++i];
        // vision text / LLM half
        } else if ((arg == "--vision-text" || arg == "--model-vision") && i + 1 < argc) {
            vision_text = argv[++i];
        // reasoning / ARLL
        } else if ((arg == "--reasoning" || arg == "--model-reasoning") && i + 1 < argc) {
            reasoning = argv[++i];
        // clinical / CSR
        } else if ((arg == "--clinical" || arg == "--model-clinical") && i + 1 < argc) {
            clinical = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--settings" && i + 1 < argc) {
            settings_path = argv[++i];
        } else if ((arg == "--temp" || arg == "--temperature") && i + 1 < argc) {
            try {
                cli_temperature = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "[cli] Invalid value for " << arg << ": " << argv[i] << '\n';
            }
        } else if ((arg == "--n-predict" || arg == "--n_predict") && i + 1 < argc) {
            try {
                cli_max_tokens = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "[cli] Invalid value for " << arg << ": " << argv[i] << '\n';
            }
        } else if (arg == "--chat-target" && i + 1 < argc) {
            const std::string target = argv[++i];
            if (target == "clinical") {
                chat_target = rmoe::ExpertTarget::Clinical;
            } else {
                chat_target = rmoe::ExpertTarget::Reasoning;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    if (image_path.empty()) {
        print_usage();
        return 1;
    }

    // ── Banner ───────────────────────────────────────────────────────────────
    rmoe::cli::print_banner();

    rmoe::initialize_backend();

    rmoe::MrTom mr_tom(rmoe::WannaStateMachine(kHardLimitIterations, kThreshold));

    // ── Load settings JSON first (base config) ───────────────────────────────
    if (!settings_path.empty()) {
        if (!mr_tom.load_settings(settings_path)) {
            std::cerr << "[settings] Failed to load settings JSON, using defaults.\n";
        }
    }

    // ── CLI flags override settings (explicit paths always win) ─────────────
    if (!vision_proj.empty() && !vision_text.empty()) {
        mr_tom.set_vision_model(vision_proj, vision_text);
    } else if (!vision_proj.empty()) {
        mr_tom.set_vision_model(vision_proj, "");
    } else if (!vision_text.empty()) {
        mr_tom.set_vision_model("", vision_text);
    }
    if (!reasoning.empty()) {
        mr_tom.set_reasoning_model(reasoning);
    }
    if (!clinical.empty()) {
        mr_tom.set_clinical_model(clinical);
    }
    if (cli_temperature >= 0.0F) {
        mr_tom.set_temperature(cli_temperature);
    }
    if (cli_max_tokens > 0) {
        mr_tom.set_max_tokens(cli_max_tokens);
    }

    // ── Patient / gate info ──────────────────────────────────────────────────
    rmoe::cli::print_input_info(image_path, kThreshold, kHardLimitIterations);

    // ── Run the pipeline ────────────────────────────────────────────────────
    const rmoe::RunSummary summary = mr_tom.process_patient_case(image_path);

    // ── Run summary + trace table ────────────────────────────────────────────
    print_run_summary(summary, kHardLimitIterations);

    // ── Clinical report ──────────────────────────────────────────────────────
    if (!summary.final_report_json.empty()) {
        print_clinical_report(summary.final_report_json);
    } else {
        std::cout << '\n';
        rmoe::cli::print_rule('=');
    }

    // ── Interactive doctor chat ──────────────────────────────────────────────
    const char* expert_label = (chat_target == rmoe::ExpertTarget::Clinical)
        ? "CSR  (clinical report)" : "ARLL (diagnostic reasoning)";
    std::cout
        << '\n'
        << rmoe::cli::kDim
        << "  Follow-up questions available  |  Expert: " << expert_label
        << "  |  Type 'exit' to quit\n"
        << rmoe::cli::kReset;
    rmoe::cli::print_rule();

    std::string doctor_query;
    while (true) {
        std::cout << '\n' << rmoe::cli::kGreen << rmoe::cli::kBold
                  << "  [DOCTOR]  " << rmoe::cli::kReset;
        std::cout.flush();
        if (!std::getline(std::cin, doctor_query)) {
            break; // EOF — exit gracefully (subprocess use)
        }
        if (doctor_query == "exit") {
            break;
        }
        if (doctor_query.empty()) {
            continue;
        }

        const std::string response = mr_tom.ask_expert(doctor_query, chat_target);
        std::cout << rmoe::cli::kCyan << "  [Mr.ToM]  "
                  << rmoe::cli::kReset << response << '\n';
    }

    std::cout << '\n';
    rmoe::cli::print_rule();
    std::cout << rmoe::cli::kDim << "  Session closed.\n" << rmoe::cli::kReset;
    rmoe::cli::print_rule();
    std::cout << '\n';

    rmoe::shutdown_backend();
    return 0;
}
