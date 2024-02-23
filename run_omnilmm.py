from pathlib import Path
from typing import Any, Literal
from omnilmm.chat import OmniLMMChat, img2base64
import json
from tqdm import tqdm
from dataset_iterator import DatasetItem, DatasetIterator


OUTPUT_JSON_PREFIX = "predictions/omni12b_jailbreak"


def build_input_message(
    dataset_item: DatasetItem,
    role: Literal["user", "assistant"] = "user",
) -> dict[str, Any]:
    # Load the image as base64
    base64_image = img2base64(dataset_item.image_path.resolve())
    # Create the input message
    input_message = json.dumps(
        [{"role": role, "content": dataset_item.prompt}],
        ensure_ascii=True,
    )
    return {"question": input_message, "image": base64_image}


if __name__ == "__main__":
    print("Loading dataset")
    dataset_iterator = DatasetIterator(
        question_csv_path=Path("data/prompt_selection.csv"),
        jailbreak_csv=Path("data/jailbreak-prompts.csv"),
        use_jailbreak_prompt=True,
        use_blank_image=False,
    )

    logger.warning(
        "Make sure the requirements for OmniLMM are installed before running this otherwise it will crash hard and fast."
    )

    print("Loading model")
    model = OmniLMMChat("openbmb/OmniLMM-12B")

    print("Running model")
    outputs = []
    for idx in tqdm(
        range(len(dataset_iterator)), desc="Running model", total=len(dataset_iterator)
    ):
        dataset_item = dataset_iterator[idx]
        input_message = build_input_message(dataset_item)
        answer = model.chat(input_message)
        assert isinstance(answer, str)

        outputs.append((dataset_item.to_key(), answer))

    print("Saving outputs")
    output_file = Path(
        f"predictions/omni12b_jailbreak{dataset_iterator.use_jailbreak_prompt}.json"
    )
    output_file.write_text(json.dumps(dict(outputs), indent=4))
