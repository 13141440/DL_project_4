# Deep Learning Coding Project 4 Report

## 1. Cover Information

- **Name:** [Your Name]
- **Student ID:** [Your Student ID]

## 2. Generative AI Usage Disclosure

I used generative AI tools to help draft and polish the report text. All final code, experiment execution, and submission decisions were checked by myself.

## 3. Custom Data Curation

I did not add extra custom training samples in the current version of this project. Therefore, the number of added samples is `0`. The submitted `custom.arrow` file should be a valid empty Arrow dataset file.

Since no additional custom data was used, no extra schema design, filtering, or cleaning pipeline was required beyond keeping the file format valid for loading during training.

## 4. Prompt and Answer Formatting

In `processors.py`, each IconQA sample is converted into a multimodal conversation that includes the query image, question text, two choice images, and a text field listing the candidate answers. The user prompt explicitly instructs the model to place its final answer inside `\boxed{}`.

For supervised fine-tuning, the training prompt uses the same structure as the evaluation prompt, while the completion is formatted as `\boxed{answer}`. This keeps the output target simple and consistent between training and evaluation.

The answer extraction strategy is also straightforward. The `extract_answer` function uses a regular expression to find the content inside `\boxed{}` and returns the extracted string after trimming whitespace. If no boxed answer is found, it returns an empty string.

This design aims to reduce ambiguity in model outputs by enforcing a clear answer format and aligning the training target with the evaluation-time instruction.

## 5. Training Configuration

The base model is `unsloth/Qwen3.5-0.8B-Base`, and the training method is supervised fine-tuning with a LoRA adapter.

In `sft_config.yaml`, I set:

- `max_length: 1024`
- `num_train_epochs: 1.0`

This configuration satisfies the assignment constraints because `max_length` is not greater than `2048`, and `num_train_epochs` is not greater than `1.0`.

I chose a relatively simple training setup as a baseline. The sequence length of `1024` is large enough for the prompt structure used here, while one training epoch keeps the training budget within the required limit.

## 6. Results

- **Validation accuracy of the base model:** [Fill in your result]
- **Validation accuracy of the trained checkpoint:** [Fill in your result]

The trained model should be compared against the base model on the validation set. In this section, you should briefly state whether fine-tuning improved performance and which design choices seemed helpful or limiting.

## 7. Conclusion

This project fine-tunes a vision-language multiple-choice QA model on IconQA using a simple boxed-answer prompting strategy and a lightweight SFT configuration. The core idea is to make the model output format explicit and easy to parse. Final effectiveness should be judged by the validation accuracy before and after training.
