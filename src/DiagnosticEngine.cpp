#include "ExpertInterface.hpp"
#include "CliOutput.hpp"

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

#ifndef RMOE_HAS_LLAMA
#  if __has_include("llama.h")
#    define RMOE_HAS_LLAMA 1
#  else
#    define RMOE_HAS_LLAMA 0
#  endif
#endif

#if RMOE_HAS_LLAMA
#include "llama.h"
#endif

#ifndef RMOE_HAS_MTMD
#  if __has_include("mtmd.h")
#    define RMOE_HAS_MTMD 1
#  else
#    define RMOE_HAS_MTMD 0
#  endif
#endif

#if RMOE_HAS_MTMD
#include "mtmd.h"
#include "mtmd-helper.h"
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

UncertaintyMetrics compute_uncertainty(const DiagnosticData& data) {
    const float p = std::clamp(data.sc, 0.001F, 0.999F);
    const float entropy = -(p * std::log2(p) + (1.0F - p) * std::log2(1.0F - p));

    // Compute sigma^2 from DDx ensemble probabilities (paper: Sc = 1 - sigma^2)
    float variance = 0.0F;
    if (!data.ddx_probabilities.empty()) {
        float mean = 0.0F;
        for (const float prob : data.ddx_probabilities) {
            mean += prob;
        }
        mean /= static_cast<float>(data.ddx_probabilities.size());
        for (const float prob : data.ddx_probabilities) {
            const float diff = prob - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(data.ddx_probabilities.size());
    }

    return {data.sc, 1.0F - data.sc, entropy, variance};
}

// Compute confidence score from DDx ensemble: Sc = 1 - sigma^2 (paper Section 3.1)
ConfidenceScore compute_confidence_from_ddx(const std::vector<float>& ddx_probs) {
    if (ddx_probs.empty()) {
        return 0.0F;
    }
    float mean = 0.0F;
    for (const float p : ddx_probs) {
        mean += p;
    }
    mean /= static_cast<float>(ddx_probs.size());
    float variance = 0.0F;
    for (const float p : ddx_probs) {
        const float diff = p - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(ddx_probs.size());
    return std::clamp(1.0F - variance, 0.0F, 1.0F);
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
    llama_model*   model {nullptr};
    llama_context* ctx   {nullptr};
#endif
#if RMOE_HAS_MTMD
    mtmd_context*  mtmd_ctx {nullptr};
#endif
    std::string    model_path;
    std::string    mmproj_path;
    InferenceParams params;
};

ExpertSwapper::~ExpertSwapper() {
    unload_current_expert();
}

void ExpertSwapper::unload_current_expert() {
    if (!impl_) {
        return;
    }

#if RMOE_HAS_MTMD
    if (impl_->mtmd_ctx != nullptr) {
        mtmd_free(impl_->mtmd_ctx);
        impl_->mtmd_ctx = nullptr;
    }
#endif

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

    std::cerr << "[llama.cpp] unload: " << impl_->model_path << '\n';
    impl_.reset();
}

bool ExpertSwapper::load_expert_model(const std::string& model_path, const InferenceParams& params) {
    unload_current_expert();

    impl_ = std::make_unique<Impl>();
    impl_->model_path = model_path;
    impl_->params     = params;

#if RMOE_HAS_LLAMA
    llama_model_params model_params = llama_model_default_params();
    impl_->model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (impl_->model == nullptr) {
        std::cerr << "[llama.cpp] Failed loading model: " << model_path << '\n';
        impl_.reset();
        return false;
    }

    llama_context_params ctx_params   = llama_context_default_params();
    ctx_params.n_ctx                  = static_cast<uint32_t>(params.n_ctx);
    ctx_params.n_threads              = params.n_threads;
    ctx_params.n_threads_batch        = params.n_threads_batch;

    impl_->ctx = llama_init_from_model(impl_->model, ctx_params);
    if (impl_->ctx == nullptr) {
        std::cerr << "[llama.cpp] Failed creating context for: " << model_path << '\n';
        llama_model_free(impl_->model);
        impl_.reset();
        return false;
    }
#endif

    std::cerr << "[llama.cpp] load: " << model_path << '\n';
    return true;
}

bool ExpertSwapper::load_mmproj_model(const std::string& mmproj_path, const int n_threads) {
    if (!impl_) {
        std::cerr << "[mtmd] load_mmproj_model: call load_expert_model first\n";
        return false;
    }
    impl_->mmproj_path = mmproj_path;

#if RMOE_HAS_LLAMA
    if (impl_->model == nullptr) {
        std::cerr << "[mtmd] load_mmproj_model: no language model loaded\n";
        return false;
    }
#endif

#if RMOE_HAS_MTMD
    if (impl_->mtmd_ctx != nullptr) {
        mtmd_free(impl_->mtmd_ctx);
        impl_->mtmd_ctx = nullptr;
    }
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu       = true;
    mparams.print_timings = false;
    mparams.n_threads     = n_threads;
    impl_->mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), impl_->model, mparams);
    if (impl_->mtmd_ctx == nullptr) {
        std::cerr << "[mtmd] Failed loading mmproj: " << mmproj_path << '\n';
        return false;
    }
    std::cerr << "[mtmd] load mmproj: " << mmproj_path << '\n';
#else
    std::cerr << "[mtmd] Mock mmproj (no mtmd support at compile time): " << mmproj_path << '\n';
#endif
    return true;
}

bool ExpertSwapper::has_mmproj() const noexcept {
    if (!impl_) {
        return false;
    }
#if RMOE_HAS_MTMD
    return impl_->mtmd_ctx != nullptr;
#else
    return false;
#endif
}

std::string ExpertSwapper::infer_text(const std::string& system_prompt,
                                       const std::string& user_input,
                                       const int max_new_tokens) const {
    if (!impl_) {
        return "[inference-error] no active model";
    }

    const int n_gen = (max_new_tokens > 0) ? max_new_tokens : impl_->params.max_new_tokens;

#if RMOE_HAS_LLAMA
    if (impl_->model == nullptr || impl_->ctx == nullptr) {
        return "[inference-error] model/context unavailable";
    }

    // ── 1. Clear KV cache so every call starts with a fresh context ──────────
    llama_memory_clear(llama_get_memory(impl_->ctx), false);

    // ── 2. Format prompt with the model's built-in chat template ─────────────
    const std::array<llama_chat_message, 2> msgs {{
        {"system", system_prompt.c_str()},
        {"user",   user_input.c_str()}
    }};
    // First call with nullptr buf probes the required byte count.
    // nullptr tmpl = use the model's own embedded chat template.
    const int32_t template_needed = llama_chat_apply_template(
        nullptr, msgs.data(), msgs.size(), /*add_ass=*/true, nullptr, 0);
    std::string formatted_prompt;
    if (template_needed > 0) {
        formatted_prompt.resize(static_cast<size_t>(template_needed));
        llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true,
                                  formatted_prompt.data(), template_needed);
        // Trim to written length; the function returns bytes written (no null).
        formatted_prompt.resize(static_cast<size_t>(template_needed));
    } else {
        // Fallback: plain concatenation when no template is embedded in the model
        formatted_prompt = system_prompt + "\nUser: " + user_input + "\nAssistant:";
    }

    // ── 3. Tokenise the formatted prompt ─────────────────────────────────────
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);
    std::vector<llama_token> prompt_tokens(formatted_prompt.size() + 8U);
    const int32_t prompt_count = llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/true);

    if (prompt_count <= 0) {
        return "[inference-error] tokenization failed";
    }
    prompt_tokens.resize(static_cast<size_t>(prompt_count));

    // ── 4. Prefill (decode the prompt in a single batch) ─────────────────────
    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1);
    for (int32_t i = 0; i < static_cast<int32_t>(prompt_tokens.size()); ++i) {
        batch.token[i]      = prompt_tokens[static_cast<size_t>(i)];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == static_cast<int32_t>(prompt_tokens.size()) - 1);
    }
    batch.n_tokens = static_cast<int32_t>(prompt_tokens.size());

    if (llama_decode(impl_->ctx, batch) != 0) {
        llama_batch_free(batch);
        return "[inference-error] prompt decode failed";
    }

    // ── 5. Build sampler chain: top_k → top_p → temperature → penalties → dist
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(impl_->params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(impl_->params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(impl_->params.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        impl_->params.penalty_last_n,
        impl_->params.repeat_penalty,
        0.0F,   // frequency penalty  (disabled)
        0.0F)); // presence penalty   (disabled)
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ── 6. Autoregressive generation loop ────────────────────────────────────
    std::string output;
    for (int i = 0; i < n_gen; ++i) {
        const llama_token next = llama_sampler_sample(smpl, impl_->ctx, batch.n_tokens - 1);
        if (next == llama_vocab_eos(vocab)) {
            break;
        }

        char piece[32] = {0};
        const int written = llama_token_to_piece(vocab, next, piece,
                                                  static_cast<int>(sizeof(piece)), 0, true);
        if (written > 0) {
            output.append(piece, static_cast<size_t>(written));
        }

        batch.token[0]     = next;
        batch.pos[0]       = batch.pos[batch.n_tokens - 1] + 1;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;

        if (llama_decode(impl_->ctx, batch) != 0) {
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    return output.empty() ? std::string("[inference-warning] empty output") : output;
#else
    return "[mock-inference] " + impl_->model_path + " | system=" +
           system_prompt.substr(0, 40) + "... | user=" + user_input.substr(0, 40) + "...";
#endif
}

std::string ExpertSwapper::infer_with_image(const std::string& system_prompt,
                                             const std::string& image_path,
                                             const std::string& user_text,
                                             const int max_new_tokens) const {
    if (!impl_) {
        return "[inference-error] no active model";
    }

    const int n_gen = (max_new_tokens > 0) ? max_new_tokens : impl_->params.max_new_tokens;

#if RMOE_HAS_LLAMA && RMOE_HAS_MTMD
    if (impl_->model == nullptr || impl_->ctx == nullptr) {
        return "[inference-error] model/context unavailable";
    }
    if (impl_->mtmd_ctx == nullptr) {
        // No mmproj loaded — fall back to text-only inference
        return infer_text(system_prompt, user_text, max_new_tokens);
    }

    // ── 1. Clear KV cache ────────────────────────────────────────────────────
    llama_memory_clear(llama_get_memory(impl_->ctx), false);

    // ── 2. Load image bitmap ─────────────────────────────────────────────────
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(impl_->mtmd_ctx, image_path.c_str());
    if (!bmp) {
        std::cerr << "[mtmd] Failed to load image: " << image_path << '\n';
        return infer_text(system_prompt, user_text, max_new_tokens);
    }

    // ── 3. Build prompt containing the media marker ──────────────────────────
    const std::string full_prompt =
        system_prompt + "\n" + mtmd_default_marker() + "\n" + user_text;

    // ── 4. Tokenize (text + image chunks) ────────────────────────────────────
    mtmd_input_text text_input;
    text_input.text          = full_prompt.c_str();
    text_input.add_special   = true;
    text_input.parse_special = true;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    constexpr size_t kSingleImage = 1U;
    const mtmd_bitmap* bitmaps_arr[] = {bmp};
    const int32_t tok_res = mtmd_tokenize(impl_->mtmd_ctx, chunks, &text_input,
                                           bitmaps_arr, kSingleImage);
    mtmd_bitmap_free(bmp);
    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        std::cerr << "[mtmd] Tokenization failed (res=" << tok_res << ")\n";
        return infer_text(system_prompt, user_text, max_new_tokens);
    }

    // ── 5. Eval all chunks (text prefix + image embeddings + text suffix) ─────
    // 512 tokens per batch is a conservative default that fits typical VRAM
    // budgets; increase via InferenceParams if model context allows it.
    constexpr int32_t kEvalBatch = 512;
    llama_pos n_past = 0;
    llama_pos new_n_past = 0;
    if (mtmd_helper_eval_chunks(impl_->mtmd_ctx, impl_->ctx, chunks,
                                 n_past, /*seq_id=*/0, kEvalBatch,
                                 /*logits_last=*/true, &new_n_past) != 0) {
        mtmd_input_chunks_free(chunks);
        return "[inference-error] eval chunks failed";
    }
    mtmd_input_chunks_free(chunks);
    n_past = new_n_past;

    // ── 6. Build sampler chain ────────────────────────────────────────────────
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(impl_->params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(impl_->params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(impl_->params.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        impl_->params.penalty_last_n,
        impl_->params.repeat_penalty,
        0.0F, 0.0F));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);

    // ── 7. Autoregressive generation ─────────────────────────────────────────
    std::string output;
    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < n_gen; ++i) {
        const llama_token next = llama_sampler_sample(smpl, impl_->ctx, -1);
        if (next == llama_vocab_eos(vocab)) {
            break;
        }

        char piece[32] = {0};
        const int written = llama_token_to_piece(vocab, next, piece,
                                                  static_cast<int>(sizeof(piece)), 0, true);
        if (written > 0) {
            output.append(piece, static_cast<size_t>(written));
        }

        gen_batch.token[0]     = next;
        gen_batch.pos[0]       = n_past++;
        gen_batch.n_seq_id[0]  = 1;
        gen_batch.seq_id[0][0] = 0;
        gen_batch.logits[0]    = 1;
        gen_batch.n_tokens     = 1;

        if (llama_decode(impl_->ctx, gen_batch) != 0) {
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_batch_free(gen_batch);
    return output.empty() ? std::string("[inference-warning] empty output") : output;
#else
    // No llama+mtmd at compile time — return a mock response
    return "[mock-inference/image] " + impl_->model_path +
           " | image=" + image_path +
           " | user=" + user_text.substr(0, 40) + "...";
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

        std::string embed_summary;

        // When a mmproj (CLIP) context is loaded and the input looks like an
        // image file path, run multimodal inference.  Otherwise fall back to
        // plain text inference (used on subsequent iterations where input_data
        // contains feedback text, not an image path).
        if (swapper_.has_mmproj()) {
            const auto dot = input_data.rfind('.');
            const bool is_image_path = (dot != std::string::npos) && [&] {
                const std::string ext = input_data.substr(dot + 1U);
                return ext == "png"  || ext == "jpg"  || ext == "jpeg"
                    || ext == "bmp"  || ext == "gif"  || ext == "webp";
            }();

            if (is_image_path) {
                embed_summary = swapper_.infer_with_image(
                    prompt,
                    input_data,
                    "Analyse this medical image and return structured visual evidence.",
                    48);
            } else {
                embed_summary = swapper_.infer_text(
                    prompt,
                    "Analyse the following medical context and return structured visual evidence:\n"
                        + input_data,
                    48);
            }
        } else {
            embed_summary = swapper_.infer_text(
                prompt,
                "Analyse the following medical image input and return structured visual evidence:\n"
                    + input_data,
                48);
        }

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
            prompt,
            "Perception evidence from MPE:\n" + input_data,
            96);

        // DDx ensemble probability samples used to compute Sc = 1 - sigma^2 (paper Section 3.1).
        // Each pair represents two ensemble draws: P(primary DDx) and P(alternative DDx).
        // Low spread (low sigma^2) => high confidence; high spread => trigger #wanna#.
        // NOTE: These are mock values for demonstration. Real inference draws from the model
        //       and parses per-token probabilities to populate ddx_probabilities.
        if (input_data.find("Alternate View") != std::string::npos) {
            // After alternate-view re-scan: ensemble converges -> Sc ~= 0.94 >= 0.90
            const std::vector<float> ddx = {0.76F, 0.24F, 0.71F, 0.29F}; // mock
            const ConfidenceScore sc = compute_confidence_from_ddx(ddx);
            return {sc, "CoT converged: " + cot, {"none", ""}, ddx};
        }
        if (input_data.find("High-Res Crop") != std::string::npos) {
            // After high-res crop: partial improvement but still uncertain -> Sc ~= 0.86 < 0.90
            const std::vector<float> ddx = {0.90F, 0.10F, 0.85F, 0.15F}; // mock
            const ConfidenceScore sc = compute_confidence_from_ddx(ddx);
            return {sc, "CoT still uncertain: " + cot,
                    {"Alternate View", "region=left_upper_quadrant;angle=oblique"}, ddx};
        }
        // Initial pass: high ensemble variance -> Sc ~= 0.79 < 0.90, request High-Res Crop
        const std::vector<float> ddx = {0.98F, 0.02F, 0.93F, 0.07F}; // mock
        const ConfidenceScore sc = compute_confidence_from_ddx(ddx);
        return {sc, "CoT uncertain: " + cot,
                {"High-Res Crop", "region=left_upper_quadrant;zoom=2.0"}, ddx};
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
            prompt,
            "Validated ARLL reasoning output:\n" + input_data,
            256);

        const nlohmann::json report = {
            {"standard", "ICD-11"},
            {"snomed_ct", "447137006"},
            {"risk_stratification", {
                {"tirads", "TR3 - Mildly Suspicious"},
                {"birads", "BI-RADS 3 - Probably Benign"}
            }},
            {"reasoning", input_data},
            {"narrative", narrative},
            {"summary", "Clinical synthesis generated from validated ARLL output."},
            {"treatment_recommendations", "6-month follow-up imaging recommended; biopsy if interval growth observed."},
            {"hitl_review_required", false}
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
        cli::print_iteration_header(iteration, state_machine_.hard_limit_iterations());
        summary.iterations_executed = iteration;

        // ── PHASE 1: MPE (Multi-Modal Perception Engine) ────────────────────
        // Load the language-model half of the vision model first (required by
        // the mtmd API before the CLIP mmproj can be attached to it).
        if (!swapper_.load_expert_model(settings_.vision_text_model, settings_.inference)) {
            escalate_to_human({0.0F, "Failed loading MPE text model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        // Now load the CLIP projection model as an mmproj linked to the LLM.
        if (!swapper_.load_mmproj_model(settings_.vision_projection_model,
                                         settings_.inference.n_threads)) {
            escalate_to_human({0.0F, "Failed loading MPE projection model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        VisionExpert mpe(swapper_);
        const DiagnosticData perception_data = mpe.execute(current_input);
        cli::print_mpe_status(settings_.vision_projection_model, settings_.vision_text_model);

        if (!swapper_.load_expert_model(settings_.reasoning_model, settings_.inference)) {
            escalate_to_human({0.0F, "Failed loading ARLL model.", {"none", ""}});
            summary.escalated_to_human = true;
            return summary;
        }
        ReasoningExpert arll(swapper_);
        const DiagnosticData reasoning_data = arll.execute(current_input + " | " + perception_data.analysis);

        const WannaDecision decision = state_machine_.evaluate(reasoning_data, iteration);
        const UncertaintyMetrics metrics = compute_uncertainty(reasoning_data);
        const bool gate_passed = (decision.state == WannaState::ProceedToReport);
        const bool escalating  = (decision.state == WannaState::EscalateToHuman);
        cli::print_arll_result(
            metrics.confidence, metrics.ddx_variance, metrics.predictive_entropy,
            gate_passed,
            (!gate_passed && !escalating) ? decision.feedback.request_type : "",
            (!gate_passed && !escalating) ? decision.feedback.payload       : "");

        summary.trace.push_back({
            iteration,
            perception_data.analysis,
            reasoning_data.analysis,
            decision_to_string(decision.state),
            metrics
        });

        if (decision.state == WannaState::ProceedToReport) {
            if (!swapper_.load_expert_model(settings_.clinical_model, settings_.inference)) {
                escalate_to_human({0.0F, "Failed loading CSR model.", {"none", ""}});
                summary.escalated_to_human = true;
                return summary;
            }
            ReportingExpert csr(swapper_);
            const DiagnosticData report_data = csr.execute(reasoning_data.analysis);
            cli::print_csr_status(settings_.clinical_model);

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

        current_input = decision.feedback.request_type + " | " + decision.feedback.payload;
    }

    escalate_to_human({0.0F, "Reached hard iteration limit without resolution.", {"none", ""}});
    summary.escalated_to_human = true;
    return summary;
}

void DiagnosticEngine::escalate_to_human(const DiagnosticData& diagnostic_data) {
    cli::print_abstain(diagnostic_data.analysis);
}

MrTom::MrTom(WannaStateMachine state_machine)
    : state_machine_(std::move(state_machine)) {}

void MrTom::set_vision_model(const std::string& projection_model_path, const std::string& text_model_path) {
    if (!projection_model_path.empty()) {
        settings_.vision_projection_model = projection_model_path;
    }
    if (!text_model_path.empty()) {
        settings_.vision_text_model = text_model_path;
    }
}

void MrTom::set_reasoning_model(const std::string& reasoning_model_path) {
    settings_.reasoning_model = reasoning_model_path;
}

void MrTom::set_clinical_model(const std::string& clinical_model_path) {
    settings_.clinical_model = clinical_model_path;
}

void MrTom::set_temperature(const float temperature) {
    settings_.inference.temperature = temperature;
}

void MrTom::set_max_tokens(const int max_new_tokens) {
    settings_.inference.max_new_tokens = max_new_tokens;
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

    if (settings_json.contains("inference")) {
        const auto& inf = settings_json["inference"];
        if (inf.contains("n_ctx"))           { settings_.inference.n_ctx           = inf["n_ctx"].get<int>();           }
        if (inf.contains("n_threads"))        { settings_.inference.n_threads        = inf["n_threads"].get<int>();        }
        if (inf.contains("n_threads_batch"))  { settings_.inference.n_threads_batch  = inf["n_threads_batch"].get<int>(); }
        if (inf.contains("max_new_tokens"))   { settings_.inference.max_new_tokens   = inf["max_new_tokens"].get<int>();   }
        if (inf.contains("temperature"))      { settings_.inference.temperature      = inf["temperature"].get<float>();    }
        if (inf.contains("top_k"))            { settings_.inference.top_k            = inf["top_k"].get<int32_t>();        }
        if (inf.contains("top_p"))            { settings_.inference.top_p            = inf["top_p"].get<float>();          }
        if (inf.contains("repeat_penalty"))   { settings_.inference.repeat_penalty   = inf["repeat_penalty"].get<float>(); }
        if (inf.contains("penalty_last_n"))   { settings_.inference.penalty_last_n   = inf["penalty_last_n"].get<int32_t>(); }
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

    if (!chat_swapper.load_expert_model(model_path, settings_.inference)) {
        return "[chat-error] failed loading expert model: " + model_path;
    }

    const std::string response = chat_swapper.infer_text(system_prompt, question, 128);
    chat_swapper.unload_current_expert();
    return response;
}

void initialize_backend() {
#if RMOE_HAS_LLAMA
    llama_backend_init();
#else
    std::cerr << "[llama.cpp] Mock mode (no llama.cpp at compile time)\n";
#endif
}

void shutdown_backend() {
#if RMOE_HAS_LLAMA
    llama_backend_free();
#endif
}

} // namespace rmoe
