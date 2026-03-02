#include "ExpertInterface.hpp"

#include <iostream>
#include <string>

namespace {
void print_usage() {
    std::cout
        << "Usage:\n"
        << "  ./rmoe_engine --vision-proj <path> --vision-text <path> --reasoning <path> --clinical <path> --image <path> [--settings <json>]\n"
        << "  Optional: --chat-target reasoning|clinical\n";
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
    rmoe::ExpertTarget chat_target = rmoe::ExpertTarget::Reasoning;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--vision-proj" && i + 1 < argc) {
            vision_proj = argv[++i];
        } else if (arg == "--vision-text" && i + 1 < argc) {
            vision_text = argv[++i];
        } else if (arg == "--reasoning" && i + 1 < argc) {
            reasoning = argv[++i];
        } else if (arg == "--clinical" && i + 1 < argc) {
            clinical = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--settings" && i + 1 < argc) {
            settings_path = argv[++i];
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

    rmoe::initialize_backend();

    rmoe::MrTom mr_tom(rmoe::WannaStateMachine(kHardLimitIterations, kThreshold));

    if (!vision_proj.empty() && !vision_text.empty()) {
        mr_tom.set_vision_model(vision_proj, vision_text);
    }
    if (!reasoning.empty()) {
        mr_tom.set_reasoning_model(reasoning);
    }
    if (!clinical.empty()) {
        mr_tom.set_clinical_model(clinical);
    }

    if (!settings_path.empty()) {
        if (!mr_tom.load_settings(settings_path)) {
            std::cerr << "[settings] Failed to load settings JSON, continuing with CLI/default values.\n";
        }
    }

    const rmoe::RunSummary summary = mr_tom.process_patient_case(image_path);
    std::cout << "[SUMMARY] success=" << summary.success
              << " escalated=" << summary.escalated_to_human
              << " iterations=" << summary.iterations_executed << '\n';

    for (const auto& t : summary.trace) {
        std::cout << "  - iter=" << t.iteration
                  << " decision=" << t.decision
                  << " Sc=" << t.metrics.confidence
                  << " H=" << t.metrics.predictive_entropy << '\n';
    }

    std::cout << "\n--- DIAGNOSIS COMPLETE ---\n";
    std::cout << "Doctor, do you have any follow-up questions? (type 'exit' to quit)\n";

    std::string doctor_query;
    while (true) {
        std::cout << "\n[DOCTOR]: ";
        std::cout.flush();
        if (!std::getline(std::cin, doctor_query)) {
            break; // EOF or stream error — exit gracefully (e.g. subprocess use)
        }
        if (doctor_query == "exit") {
            break;
        }
        if (doctor_query.empty()) {
            continue;
        }

        const std::string response = mr_tom.ask_expert(doctor_query, chat_target);
        std::cout << "[Mr.ToM]: " << response << '\n';
    }

    rmoe::shutdown_backend();
    return 0;
}
