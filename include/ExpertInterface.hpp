#pragma once

#include <memory>
#include <string>
#include <vector>

namespace rmoe {

using ConfidenceScore = float;

enum class ExpertTarget {
    Reasoning,
    Clinical
};

struct FeedbackTensor {
    std::string request_type;
    std::string payload;
};

// Inference hyperparameters forwarded to llama.cpp context and sampler chain.
// Populated from CLI defaults, settings JSON, or per-expert overrides.
struct InferenceParams {
    int     n_ctx           {4096}; // KV-cache context window (tokens)
    int     n_threads       {4};    // decode threads
    int     n_threads_batch {4};    // prompt-eval threads
    int     max_new_tokens  {128};  // default generation budget (-1 = use this value)
    float   temperature     {0.2F}; // sampling temperature (paper: 0.2 for clinical precision)
    int32_t top_k           {40};   // top-k filter (paper CLI: --top_k 40)
    float   top_p           {0.95F};// nucleus probability threshold
    float   repeat_penalty  {1.1F}; // repetition penalty multiplier
    int32_t penalty_last_n  {64};   // window of tokens considered for repetition penalty
};

struct DiagnosticData {
    ConfidenceScore sc {0.0F};
    std::string analysis;
    FeedbackTensor feedback {};
    // DDx ensemble probability samples used by ARLL to compute Sc = 1 - sigma^2
    std::vector<float> ddx_probabilities;
};

struct UncertaintyMetrics {
    ConfidenceScore confidence {0.0F};
    float uncertainty {1.0F};
    float predictive_entropy {0.0F};
    float ddx_variance {0.0F}; // sigma^2 in Sc = 1 - sigma^2 (ensemble DDx variance)
};

struct IterationTrace {
    int iteration {1};
    std::string perception_summary;
    std::string reasoning_summary;
    std::string decision;
    UncertaintyMetrics metrics {};
};

struct RunSummary {
    bool success {false};
    bool escalated_to_human {false};
    int iterations_executed {0};
    std::string final_report_json;
    std::vector<IterationTrace> trace;
};

struct ModelSettings {
    std::string vision_projection_model {"models/vision_proj.gguf"};
    std::string vision_text_model       {"models/vision_text.gguf"};
    std::string reasoning_model         {"models/reasoning_expert.gguf"};
    std::string clinical_model          {"models/clinical_expert.gguf"};
    InferenceParams inference           {};
};

class IClinicalExpert {
public:
    virtual ~IClinicalExpert() = default;
    virtual DiagnosticData execute(const std::string& input_data) = 0;
    [[nodiscard]] virtual std::string get_expert_name() const = 0;
};

class IPerceptionExpert : public IClinicalExpert {
public:
    ~IPerceptionExpert() override = default;
};

class IReasoningExpert : public IClinicalExpert {
public:
    ~IReasoningExpert() override = default;
};

class IReportingExpert : public IClinicalExpert {
public:
    ~IReportingExpert() override = default;
};

enum class WannaState {
    ProceedToReport,
    RequestHighResCrop,
    RequestAlternateView,
    EscalateToHuman
};

struct WannaDecision {
    WannaState state {WannaState::ProceedToReport};
    int iteration {1};
    FeedbackTensor feedback {};
};

class WannaStateMachine {
public:
    WannaStateMachine(int hard_limit_iterations, ConfidenceScore threshold);

    [[nodiscard]] WannaDecision evaluate(const DiagnosticData& reasoning_result, int iteration) const;
    [[nodiscard]] int hard_limit_iterations() const noexcept;
    [[nodiscard]] ConfidenceScore threshold() const noexcept;

private:
    int hard_limit_iterations_;
    ConfidenceScore threshold_;
};

class ExpertSwapper {
public:
    ExpertSwapper() = default;
    ~ExpertSwapper();

    void unload_current_expert();
    bool load_expert_model(const std::string& model_path, const InferenceParams& params = {});

    // Loads a CLIP mmproj model that is linked to the already-loaded language model.
    // Must be called after a successful load_expert_model().
    bool load_mmproj_model(const std::string& mmproj_path, int n_threads = 4);

    // Returns true when a CLIP mmproj model is currently loaded.
    [[nodiscard]] bool has_mmproj() const noexcept;

    // Infer using a system prompt and user input.
    // Applies the model's chat template (llama mode) or returns a mock response.
    // max_new_tokens overrides InferenceParams::max_new_tokens when > 0.
    [[nodiscard]] std::string infer_text(const std::string& system_prompt,
                                         const std::string& user_input,
                                         int max_new_tokens = -1) const;

    // Run multimodal inference: embeds image_path via the mmproj CLIP model,
    // then generates text with the language model.
    // Falls back to infer_text when no mmproj context is loaded.
    [[nodiscard]] std::string infer_with_image(const std::string& system_prompt,
                                               const std::string& image_path,
                                               const std::string& user_text,
                                               int max_new_tokens = -1) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DiagnosticEngine {
public:
    explicit DiagnosticEngine(WannaStateMachine state_machine, ModelSettings settings = {});
    RunSummary run_diagnostics(const std::string& patient_input);

private:
    static void escalate_to_human(const DiagnosticData& diagnostic_data);

    WannaStateMachine state_machine_;
    ModelSettings settings_;
    ExpertSwapper swapper_;
};

class MrTom {
public:
    explicit MrTom(WannaStateMachine state_machine);

    void set_vision_model(const std::string& projection_model_path, const std::string& text_model_path);
    void set_reasoning_model(const std::string& reasoning_model_path);
    void set_clinical_model(const std::string& clinical_model_path);
    void configure_gate(int max_iterations, ConfidenceScore threshold);
    bool load_settings(const std::string& settings_json_path);

    RunSummary process_patient_case(const std::string& patient_input) const;
    std::string ask_expert(const std::string& question, ExpertTarget target = ExpertTarget::Reasoning) const;

    void set_temperature(float temperature);
    void set_max_tokens(int max_new_tokens);

private:
    WannaStateMachine state_machine_;
    ModelSettings settings_;
};

void initialize_backend();
void shutdown_backend();

} // namespace rmoe
