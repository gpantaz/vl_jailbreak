from pathlib import Path
from typing import Literal, TypedDict
from omnilmm.chat import img2base64, OmniLMMChat
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

OUTPUT_CSV_FILE = "omnilmm_output.csv"


class OmniLMMInput(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    image: bytes


class OutputRow(TypedDict):
    question: str
    iamge_path: str
    answer: str


def build_input_message(
    content: str, image_path: Path, role: Literal["user", "assistant"] = "user"
) -> OmniLMMInput:
    # Load the image as base64
    base64_image = img2base64(image_path)
    # Create the input message
    input_message = json.dumps(
        [{"role": role, "content": content}],
        ensure_ascii=True,
    )
    return {"role": role, "content": input_message, "image": base64_image}


if __name__ == "__main__":
    logger.warning(
        "Make sure the requirements for OmniLMM are installed before running this otherwise it will crash hard and fast."
    )

    model = OmniLMMChat("openbmb/OmniLMM-12B")

    # TODO: Load the image-question pairs
    image_text_pairs: list[tuple[str, Path]] = []

    outputs = []
    for question, image_path in tqdm(
        image_text_pairs, desc="Running model", total=len(image_text_pairs)
    ):
        input_message = build_input_message(question, image_path)
        answer = model.chat(input_message)
        output_row = {
            "question": question,
            "image_path": str(image_path),
            "answer": answer,
        }
        outputs.append(output_row)

    # Save the outputs to a CSV file
    with open(OUTPUT_CSV_FILE, "w") as f:
        f.write("question,image_path,answer\n")
        for row in tqdm(outputs, desc="Saving outputs", total=len(outputs)):
            f.write(f"{row['question']},{row['image_path']},{row['answer']}\n")

    logger.info(f"Outputs saved to {OUTPUT_CSV_FILE}")
