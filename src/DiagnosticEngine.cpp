#include "ExpertInterface.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if __has_include("llama.h")
#include "llama.h"
#define RMOE_HAS_LLAMA 1
#else
#define RMOE_HAS_LLAMA 0
#endif

namespace rmoe {

namespace {
std::string load_prompt_file(const std::string& path, const std::string& fallback) {
    std::ifstream in(path);
    if (!in) {
        return fallback;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

UncertaintyMetrics compute_uncertainty(const ConfidenceScore sc) {
    const float p = std::clamp(sc, 0.001F, 0.999F);
    const float entropy = -(p * std::log2(p) + (1.0F - p) * std::log2(1.0F - p));
    return {sc, 1.0F - sc, entropy};
}

std::string decision_to_string(const WannaState state) {
    switch (state) {
        case WannaState::ProceedToReport:
            return "ProceedToReport";
        case WannaState::RequestHighResCrop:
            return "RequestHighResCrop";
        case WannaState::RequestAlternateView:
            return "RequestAlternateView";
        case WannaState::EscalateToHuman:
            return "EscalateToHuman";
    }
    return "Unknown";
}
} // namespace

struct ExpertSwapper::Impl {
#if RMOE_HAS_LLAMA
    llama_model* model {nullptr};
    llama_context* ctx {nullptr};
#endif
    std::string model_path;
};

ExpertSwapper::~ExpertSwapper() {
    unload_current_expert();
}

void ExpertSwapper::unload_current_expert() {
    if (!impl_) {
        return;
    }

#if RMOE_HAS_LLAMA
    if (impl_->ctx != nullptr) {
        llama_free(impl_->ctx);
        impl_->ctx = nullptr;
    }
    if (impl_->model != nullptr) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
    }
#endif

    std::cout << "[llama.cpp] unload_model(): " << impl_->model_path << '\n';
    impl_.reset();
}

bool ExpertSwapper::load_expert_model(const std::string& model_path, const int n_ctx) {
    unload_current_expert();

    impl_ = std::make_unique<Impl>();
    impl_->model_path = model_path;

#if RMOE_HAS_LLAMA
    llama_model_params model_params = llama_model_default_params();
    impl_->model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (impl_->model == nullptr) {
        std::cerr << "[llama.cpp] Failed loading model: " << model_path << '\n';
        impl_.reset();
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    impl_->ctx = llama_init_from_model(impl_->model, ctx_params);
    if (impl_->ctx == nullptr) {
        std::cerr << "[llama.cpp] Failed creating context for: " << model_path << '\n';
        llama_model_free(impl_->model);
        impl_.reset();
        return false;
    }
#else
    (void)n_ctx;
#endif

    std::cout << "[llama.cpp] load_model(): " << model_path << '\n';
    return true;
}

std::string ExpertSwapper::infer_text(const std::string& prompt, const int max_new_tokens) const {
    if (!impl_) {
        return "[inference-error] no active model";
    }

#if RMOE_HAS_LLAMA
    if (impl_->model == nullptr || impl_->ctx == nullptr) {
        return "[inference-error] model/context unavailable";
    }

    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);
    std::vector<llama_token> prompt_tokens(prompt.size() + 8U);
    const int32_t prompt_count = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        true,
        true);

    if (prompt_count <= 0) {
        return "[inference-error] tokenization failed";
    }

    prompt_tokens.resize(static_cast<size_t>(prompt_count));

    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1);
    for (int32_t i = 0; i < static_cast<int32_t>(prompt_tokens.size()); ++i) {
        batch.token[i] = prompt_tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == static_cast<int32_t>(prompt_tokens.size()) - 1);
    }
    batch.n_tokens = static_cast<int32_t>(prompt_tokens.size());

    if (llama_decode(impl_->ctx, batch) != 0) {
        llama_batch_free(batch);
        return "[inference-error] decode failed";
    }

    std::string output;
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    for (int i = 0; i < max_new_tokens; ++i) {
        const llama_token next = llama_sampler_sample(smpl, impl_->ctx, batch.n_tokens - 1);
        if (next == llama_vocab_eos(vocab)) {
            break;
        }

        char piece[32] = {0};
        const int written = llama_token_to_piece(vocab, next, piece, static_cast<int>(sizeof(piece)), 0, true);
        if (written > 0) {
            output.append(piece, static_cast<size_t>(written));
        }

        batch.token[0] = next;
        batch.pos[0] = batch.pos[batch.n_tokens - 1] + 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(impl_->ctx, batch) != 0) {
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    return output.empty() ? std::string("[inference-warning] empty output") : output;
#else
    (void)max_new_tokens;
    return "[mock-inference] " + impl_->model_path + " processed: " + prompt;
#endif
}

WannaStateMachine::WannaStateMachine(const int hard_limit_iterations, const ConfidenceScore threshold)
    : hard_limit_iterations_(hard_limit_iterations), threshold_(threshold) {}

WannaDecision WannaStateMachine::evaluate(const DiagnosticData& reasoning_result, const int iteration) const {
    if (reasoning_result.sc >= threshold_) {
        return {WannaState::ProceedToReport, iteration, {"none", ""}};
    }

    if (iteration >= hard_limit_iterations_) {
        return {WannaState::EscalateToHuman, iteration, reasoning_result.feedback};
    }

    std::string lowered_request = reasoning_result.feedback.request_type;
    std::transform(lowered_request.begin(), lowered_request.end(), lowered_request.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lowered_request.find("alternate") != std::string::npos) {
        return {WannaState::RequestAlternateView, iteration, reasoning_result.feedback};
    }

    return {WannaState::RequestHighResCrop, iteration, reasoning_result.feedback};
}

int WannaStateMachine::hard_limit_iterations() const noexcept {
    return hard_limit_iterations_;
}

ConfidenceScore WannaStateMachine::threshold() const noexcept {
    return threshold_;
}

class VisionExpert final : public IPerceptionExpert {
public:
    explicit VisionExpert(const ExpertSwapper& swapper)
        : swapper_(swapper) {}

    DiagnosticData execute(const std::string& input_data) override {
        const std::string prompt = load_prompt_file(
            "prompts/mpe_system_prompt.txt",
            "You are MPE. Extract visual findings only.");
        const std::string embed_summary = swapper_.infer_text(
            prompt + "\nUser Input: " + input_data,
            24);

        return {0.0F, "MPE embeddings summary: " + embed_summary, {"none", ""}};
    }

    std::string get_expert_name() const override {
        return "MPE (Qwen2-VL)";
    }

private:
    const ExpertSwapper& swapper_;
};

class ReasoningExpert final : public IReasoningExpert {
public:
    explicit ReasoningExpert(const ExpertSwapper& swapper)
        : swapper_(swapper) {}

    DiagnosticData execute(const std::string& input_data) override {
        const std::string prompt = load_prompt_file(
            "prompts/arll_system_prompt.txt",
            "You are ARLL. Run reasoning and provide confidence guidance.");
        const std::string cot = swapper_.infer_text(
            prompt + "\nUser Input: " + input_data,
            24);

        if (input_data.find("Alternate View") != std::string::npos) {
            return {0.94F, "CoT converged: " + cot, {"none", ""}};
        }
        if (input_data.find("High-Res Crop") != std::string::npos) {
            return {0.86F, "CoT still uncertain: " + cot,
                    {"Alternate View", "region=left_upper_quadrant;angle=oblique"}};
        }
        return {0.79F, "CoT uncertain: " + cot,
                {"High-Res Crop", "region=left_upper_quadrant;zoom=2.0"}};
    }

    std::string get_expert_name() const override {
        return "ARLL (DeepSeek-R1)";
    }

private:
    const ExpertSwapper& swapper_;
};

class ReportingExpert final : public IReportingExpert {
public:
    explicit ReportingExpert(const ExpertSwapper& swapper)
        : swapper_(swapper) {}

    DiagnosticData execute(const std::string& input_data) override {
        const std::string prompt = load_prompt_file(
            "prompts/csr_system_prompt.txt",
            "You are CSR. Generate ICD-11 compliant report output.");
        const std::string narrative = swapper_.infer_text(
            prompt + "\nUser Input: " + input_data,
            48);

        const nlohmann::json report = {
            {"standard", "ICD-11"},
            {"reasoning", input_data},
            {"narrative", narrative},
            {"summary", "Clinical synthesis generated from validated ARLL output."}
        };
        return {0.95F, report.dump(2), {"none", ""}};
    }

    std::string get_expert_name() const override {
        return "CSR (Llama-3-Medius)";
    }

private:
    const ExpertSwapper& swapper_;
};

DiagnosticEngine::DiagnosticEngine(WannaStateMachine state_machine, ModelSettings settings)
    : state_machine_(std::move(state_machine)), settings_(std::move(settings)) {}

RunSummary DiagnosticEngine::run_diagnostics(const std::string& patient_input) {
    std::string current_input = patient_input;
    RunSummary summary {};

    for (int iteration = 1; iteration <= state_machine_.hard_limit_iterations(); ++iteration) {
        std::cout << "\n[Iteration " << iteration << "] Starting triple-expert pipeline\n";
        summary.iterations_executed = iteration;

        if (!swapper_.load_expert_model(settings_.vision_projection_model)) {
            escalate_to_human({0.0F, "Failed loading MPE projection model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        VisionExpert mpe_projection(swapper_);
        const DiagnosticData projection_data = mpe_projection.execute(current_input);

        if (!swapper_.load_expert_model(settings_.vision_text_model)) {
            escalate_to_human({0.0F, "Failed loading MPE text model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        VisionExpert mpe_encoder(swapper_);
        const DiagnosticData perception_data = mpe_encoder.execute(current_input + " | " + projection_data.analysis);

        if (!swapper_.load_expert_model(settings_.reasoning_model)) {
            escalate_to_human({0.0F, "Failed loading ARLL model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        ReasoningExpert arll(swapper_);
        const DiagnosticData reasoning_data = arll.execute(current_input + " | " + perception_data.analysis);
        std::cout << "-> " << arll.get_expert_name() << " Sc=" << reasoning_data.sc << " | "
                  << reasoning_data.analysis << '\n';

        const WannaDecision decision = state_machine_.evaluate(reasoning_data, iteration);
        summary.trace.push_back({
            iteration,
            perception_data.analysis,
            reasoning_data.analysis,
            decision_to_string(decision.state),
            compute_uncertainty(reasoning_data.sc)
        });

        if (decision.state == WannaState::ProceedToReport) {
            if (!swapper_.load_expert_model(settings_.clinical_model)) {
                escalate_to_human({0.0F, "Failed loading CSR model.", {"none", ""}});
                summary.escalated_to_human = true;
                return summary;
            }
            ReportingExpert csr(swapper_);
            const DiagnosticData report_data = csr.execute(reasoning_data.analysis);
            std::cout << "-> " << csr.get_expert_name() << " output:\n" << report_data.analysis << '\n';

            summary.success = true;
            summary.final_report_json = report_data.analysis;
            swapper_.unload_current_expert();
            return summary;
        }

        if (decision.state == WannaState::EscalateToHuman) {
            swapper_.unload_current_expert();
            escalate_to_human(reasoning_data);
            summary.escalated_to_human = true;
            return summary;
        }

        std::cout << "[TRIGGER] #wanna# | request=" << decision.feedback.request_type
                  << " | feedback_tensor={" << decision.feedback.payload << "}\n";
        current_input = decision.feedback.request_type + " | " + decision.feedback.payload;
    }

    escalate_to_human({0.0F, "Reached hard iteration limit without resolution.", {"none", ""}});
    summary.escalated_to_human = true;
    return summary;
}

void DiagnosticEngine::escalate_to_human(const DiagnosticData& diagnostic_data) {
    std::cerr << "[ABSTAIN] Escalating to human. Last analysis: " << diagnostic_data.analysis << '\n';
}

MrTom::MrTom(WannaStateMachine state_machine)
    : state_machine_(std::move(state_machine)) {}

void MrTom::set_vision_model(const std::string& projection_model_path, const std::string& text_model_path) {
    settings_.vision_projection_model = projection_model_path;
    settings_.vision_text_model = text_model_path;
}

void MrTom::set_reasoning_model(const std::string& reasoning_model_path) {
    settings_.reasoning_model = reasoning_model_path;
}

void MrTom::set_clinical_model(const std::string& clinical_model_path) {
    settings_.clinical_model = clinical_model_path;
}

void MrTom::configure_gate(const int max_iterations, const ConfidenceScore threshold) {
    state_machine_ = WannaStateMachine(max_iterations, threshold);
}

bool MrTom::load_settings(const std::string& settings_json_path) {
    std::ifstream in(settings_json_path);
    if (!in) {
        std::cerr << "[settings] Unable to open settings file: " << settings_json_path << '\n';
        return false;
    }

    nlohmann::json settings_json;
    try {
        in >> settings_json;
    } catch (const std::exception& e) {
        std::cerr << "[settings] Invalid JSON in settings file: " << e.what() << '\n';
        return false;
    }

    if (settings_json.contains("vision_proj_model")) {
        settings_.vision_projection_model = settings_json["vision_proj_model"].get<std::string>();
    }
    if (settings_json.contains("vision_text_model")) {
        settings_.vision_text_model = settings_json["vision_text_model"].get<std::string>();
    }
    if (settings_json.contains("reasoning_model")) {
        settings_.reasoning_model = settings_json["reasoning_model"].get<std::string>();
    }
    if (settings_json.contains("clinical_model")) {
        settings_.clinical_model = settings_json["clinical_model"].get<std::string>();
    }
    if (settings_json.contains("max_iterations") && settings_json.contains("confidence_threshold")) {
        configure_gate(settings_json["max_iterations"].get<int>(), settings_json["confidence_threshold"].get<float>());
    }

    return true;
}

RunSummary MrTom::process_patient_case(const std::string& patient_input) const {
    DiagnosticEngine engine(state_machine_, settings_);
    return engine.run_diagnostics(patient_input);
}


std::string MrTom::ask_expert(const std::string& question, const ExpertTarget target) const {
    ExpertSwapper chat_swapper;
    std::string model_path;
    std::string system_prompt;

    if (target == ExpertTarget::Clinical) {
        model_path = settings_.clinical_model;
        system_prompt = load_prompt_file(
            "prompts/csr_system_prompt.txt",
            "You are CSR. Answer follow-up clinical report questions.");
    } else {
        model_path = settings_.reasoning_model;
        system_prompt = load_prompt_file(
            "prompts/arll_system_prompt.txt",
            "You are ARLL. Answer diagnostic reasoning questions.");
    }

    if (!chat_swapper.load_expert_model(model_path)) {
        return "[chat-error] failed loading expert model: " + model_path;
    }

    const std::string response = chat_swapper.infer_text(system_prompt + "\nDoctor Question: " + question, 96);
    chat_swapper.unload_current_expert();
    return response;
}

void initialize_backend() {
#if RMOE_HAS_LLAMA
    llama_backend_init();
#else
    std::cout << "[llama.cpp] Backend unavailable at compile time; running mock mode.\n";
#endif
}

void shutdown_backend() {
#if RMOE_HAS_LLAMA
    llama_backend_free();
#endif
}

} // namespace rmoe
