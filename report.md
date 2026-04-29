# Deep Learning Coding Project 4 Report

## 1. Cover Information

- **Name:** Wangqiantong
- **Student ID:** 2024011305

## 2. Generative AI Usage Disclosure

I used Claude (Anthropic) to assist with designing the prompt template structure, selecting SFT hyperparameters, and drafting the report text. All final code, experiment execution, and submission decisions were reviewed and confirmed by myself.

## 3. Custom Data Curation

I did not add extra custom training samples in this project. The submitted `custom.arrow` is a valid empty Arrow dataset file with 0 samples.

- **Schema:** Same as the provided IconQA dataset (`question`, `choices`, `answer`, `query_image`, `choice_image_0`, `choice_image_1`), all with appropriate types (string for text fields, Image for image fields).
- **Data source:** N/A
- **Filtering/cleaning:** N/A
- **Number of samples added:** 0

The `convert_custom_train_to_conversation` function reuses the same conversion logic as `convert_icon_qa_train_to_conversation`, so if custom data were added with the same schema, it would be processed identically.

## 4. Prompt and Answer Formatting

### Prompt Design

Each IconQA sample is converted into a single-turn user message with multimodal content:

1. The **query image** is placed first so the model sees the visual context before the question.
2. The **question text** follows immediately.
3. **Choice A** and **Choice B** images are presented with concise labels ("A:" and "B:").
4. A short **instruction** at the end tells the model to answer with `\boxed{A}` or `\boxed{B}`.

The instruction is kept minimal: "Look at the image and the two choices below. Which choice answers the question? Answer with \boxed{A} or \boxed{B}." This reduces token consumption while remaining unambiguous for the model.

### Answer Format

- During training, the assistant completion is simply `\boxed{A}` or `\boxed{B}` (one token pattern).
- The `\boxed{}` format provides a clear, parseable boundary for answer extraction.

### Answer Extraction (`extract_answer`)

The extraction uses a multi-level fallback strategy:

1. Search for `\boxed{...}` pattern via regex.
2. If found, try to parse the content as a letter (A/B) or digit (0/1).
3. If no boxed answer, fall back to searching the raw text for letter or digit patterns.
4. If all parsing fails, return the raw text (which will be marked incorrect rather than crashing).

This robust extraction handles cases where the model outputs extra text, uses lowercase, or outputs numeric indices instead of letters.

## 5. Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `max_length` | 1024 | Token analysis shows prompts average ~355 tokens (max ~574), so 1024 provides sufficient headroom |
| `per_device_train_batch_size` | 2 | Small batch to fit in GPU memory with vision tokens |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8, balancing stability and budget |
| `max_steps` | 250 | Total budget = 2 x 4 x 250 = 2000 (maximum allowed) |
| `learning_rate` | 2e-4 | Higher than default (2e-5) since LoRA adapters benefit from larger LR |
| `lr_scheduler_type` | cosine | Smooth decay helps convergence for short training runs |
| `warmup_ratio` | 0.1 | 25 warmup steps to stabilize early training |
| `weight_decay` | 0.01 | Light regularization to prevent overfitting |
| `bf16` | true | Required for Qwen3.5 (float16 causes NaN gradients) |
| `max_grad_norm` | 1.0 | Standard gradient clipping |
| `save_strategy` | no | Only save final checkpoint to avoid disk overhead |

The LoRA adapter uses unsloth's defaults: rank=16, alpha=16, targeting all linear layers in both vision and language components.

## 6. Results

- **Validation accuracy of the base model (zero-shot):** [TO BE FILLED AFTER GPU EVALUATION]
- **Validation accuracy of the trained checkpoint:** [TO BE FILLED AFTER GPU EVALUATION]

### Discussion

[TO BE FILLED AFTER EXPERIMENTS]

Key design choices expected to help performance:
- Placing the query image first gives the model visual context before processing the question.
- The concise `\boxed{}` answer format reduces the completion length, making the training signal cleaner.
- Using `max_steps=250` with effective batch size 8 ensures the model sees the full 1000-sample training set approximately 2 times within the budget constraint.
- A higher learning rate (2e-4) is appropriate for LoRA fine-tuning where only adapter weights are updated.
- Cosine scheduling with warmup provides stable training dynamics for short runs.
