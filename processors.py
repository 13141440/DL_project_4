import re
from typing import Any, TypedDict

from PIL.Image import Image

type Conversation = list[dict[str, Any]]


class ConversationalLanguageModeling(TypedDict):
    messages: Conversation


class ConversationalPromptCompletion(TypedDict):
    prompt: Conversation
    completion: Conversation


class IconQASample(TypedDict):
    question: str
    choices: str
    answer: str | None
    query_image: Image
    choice_image_0: Image
    choice_image_1: Image


# IconQA stores answers as filenames such as "choice_0.png" / "choice_1.png".
# We let the model answer with simple letters (A / B) and translate them back
# to the filename format expected by the evaluator.
_LETTER_TO_FILENAME = {"A": "choice_0.png", "B": "choice_1.png"}
_FILENAME_TO_LETTER = {v: k for k, v in _LETTER_TO_FILENAME.items()}


_INSTRUCTION = (
    "You will see a reference image and two candidate images labelled A and B. "
    "Decide which candidate image best answers the question. "
    "Reply with only the chosen letter inside \\boxed{}, "
    "either \\boxed{A} or \\boxed{B}. Do not output anything else."
)


def _build_user_content(sample: IconQASample) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": _INSTRUCTION},
        {"type": "text", "text": f"Question: {sample['question']}"},
        {"type": "text", "text": "Reference image:"},
        {"type": "image", "image": sample["query_image"]},
        {"type": "text", "text": "Choice A:"},
        {"type": "image", "image": sample["choice_image_0"]},
        {"type": "text", "text": "Choice B:"},
        {"type": "image", "image": sample["choice_image_1"]},
        {
            "type": "text",
            "text": "Final answer (only \\boxed{A} or \\boxed{B}):",
        },
    ]


def convert_custom_train_to_conversation(
    sample: dict[str, Any],
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from a custom training sample.

    Args:
        sample: A sample in the custom training dataset. The schema of this
            dataset is student-defined.

    Returns:
        A conversation for training. You are responsible for converting your
        custom sample format into this prompt-completion structure.
    """

    # YOUR CODE BEGIN.

    icon_qa_sample = IconQASample(
        question=sample["question"],
        choices=sample.get("choices", "choice_0.png,choice_1.png"),
        answer=sample["answer"],
        query_image=sample["query_image"],
        choice_image_0=sample["choice_image_0"],
        choice_image_1=sample["choice_image_1"],
    )
    return convert_icon_qa_train_to_conversation(icon_qa_sample)

    # YOUR CODE END.


def convert_icon_qa_test_to_conversation(
    sample: IconQASample,
) -> ConversationalLanguageModeling:
    """Builds one eval conversation from an IconQA sample.

    Args:
        sample: A IconQA sample, whose ``answer`` field is always ``None``.

    Returns:
        A conversation for testing.
    """

    # YOUR CODE BEGIN.

    return ConversationalLanguageModeling(
        messages=[
            {
                "role": "user",
                "content": _build_user_content(sample),
            }
        ]
    )

    # YOUR CODE END.


def convert_icon_qa_train_to_conversation(
    sample: IconQASample,
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from an IconQA training sample.

    Args:
        sample: A IconQA sample.

    Returns:
        A conversation for training, where the prompt is the same as the test conversation
    """

    # YOUR CODE BEGIN.

    answer = sample["answer"]
    letter = _FILENAME_TO_LETTER.get(answer, answer)

    return ConversationalPromptCompletion(
        prompt=convert_icon_qa_test_to_conversation(sample)["messages"],
        completion=[
            {
                "role": "assistant",
                "content": f"\\boxed{{{letter}}}",
            }
        ],
    )

    # YOUR CODE END.


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]*?)\s*\}")
_LETTER_RE = re.compile(r"\b([AB])\b")
_DIGIT_RE = re.compile(r"\b([01])\b")


def extract_answer(generated_text: str) -> str:
    """Extracts the final answer token from model output.

    Args:
        generated_text: Raw generated text.

    Returns:
        The parsed answer in the same string format as the dataset's ``answer``
        field (``"choice_0.png"`` or ``"choice_1.png"``). If parsing fails, the
        raw stripped output is returned so that it will be marked as incorrect
        rather than crashing the evaluator.
    """

    # YOUR CODE BEGIN.

    text = generated_text.strip()

    boxed = _BOXED_RE.search(text)
    candidates = []
    if boxed:
        candidates.append(boxed.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        normalized = candidate.strip().strip(".")

        if normalized in _FILENAME_TO_LETTER:
            return normalized

        upper = normalized.upper()
        if upper in _LETTER_TO_FILENAME:
            return _LETTER_TO_FILENAME[upper]

        if normalized in {"0", "1"}:
            return f"choice_{normalized}.png"

        letter_match = _LETTER_RE.search(upper)
        if letter_match:
            return _LETTER_TO_FILENAME[letter_match.group(1)]

        digit_match = _DIGIT_RE.search(normalized)
        if digit_match:
            return f"choice_{digit_match.group(1)}.png"

    return text

    # YOUR CODE END.
