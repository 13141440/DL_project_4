# Deep Learning Coding Project 4 Report

## 1. Cover Information

- **Name:** Wangqiantong
- **Student ID:** 2024011305

## 2. Generative AI Usage Disclosure

I used Claude (Anthropic) and Cursor to assist with designing the prompt template structure and drafting the report text. All final code, experiment execution, and submission decisions were reviewed and confirmed by myself.

## 3. Custom Data Curation

I constructed `custom.arrow` with 1000 samples using the following pipeline:

- **Data source:** `lmms-lab/ICON-QA` dataset (val split) from HuggingFace, which contains 21,488 IconQA samples.
- **Schema:** Same as the provided IconQA dataset (`question`, `choices`, `answer`, `query_image`, `choice_image_0`, `choice_image_1`), all with appropriate types (string for text fields, Image for image fields).

**Filtering/cleaning steps:**

1. Filtered for `choose_img` question type only (11,534 samples).
2. Removed samples whose `question_id` overlaps with the provided train/val sets (remaining: 10,334).
3. Kept samples that are binary-choice (`choices == "choice_0.png,choice_1.png"`): 568 samples.
4. For multi-choice samples (4+ options), kept those whose answer is `choice_0.png` or `choice_1.png` (i.e., the correct answer image is available in the dataset columns), converted them to binary-choice format: 224 additional samples.
5. Total after steps 3-4: 792 samples.

**Data augmentation:**

6. To fill the remaining 208 slots (reaching the 1000-sample cap), I applied A/B swap augmentation: randomly selected 208 samples, swapped `choice_image_0` and `choice_image_1`, and flipped the answer accordingly. This teaches the model to judge by image content rather than position bias.

- **Number of samples added:** 1000 (792 real + 208 augmented)
- **Final answer distribution:** choice_0.png: 488, choice_1.png: 512 (balanced)

The `convert_custom_train_to_conversation` function reuses the same conversion logic as `convert_icon_qa_train_to_conversation`, since the custom data shares the same schema.

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

- **Validation accuracy of the base model (zero-shot):** 0.735
- **Validation accuracy of the trained checkpoint (without data cleaning):** 0.885
- **Validation accuracy of the trained checkpoint (with data cleaning + augmentation):** 0.925

### Discussion

| Configuration | Accuracy |
|---|---|
| Base model (zero-shot) | 0.735 |
| Trained with raw custom data (includes incompatible 4-choice samples) | 0.885 |
| Trained with cleaned custom data (filtered to 2-choice + AB-swap augmentation) | 0.925 |

Fine-tuning improved accuracy from 0.735 to 0.925 (+19.0 percentage points). Data cleaning and augmentation provided an additional +4.0 points over the raw data version, confirming that data quality matters more than quantity.

Key design choices that helped performance:
- **Data cleaning:** Removing incompatible 4-choice samples (whose answers reference non-existent choice images) eliminated noisy training signals.
- **AB-swap augmentation:** Teaching the model to judge by image content rather than position bias improved generalization.
- Placing the query image first gives the model visual context before processing the question.
- The concise `\boxed{}` answer format reduces the completion length, making the training signal cleaner.
- Adding 1000 custom training samples from the IconQA dataset doubled the effective training data, providing more diverse visual reasoning examples.
- Using `max_steps=250` with effective batch size 8 maximizes the allowed training budget (2000 total).
- A higher learning rate (2e-4) is appropriate for LoRA fine-tuning where only adapter weights are updated.
- Cosine scheduling with warmup provides stable training dynamics for short runs.
