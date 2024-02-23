from pathlib import Path
import json
from tqdm import tqdm
from dataset_iterator import DatasetIterator, DatasetItem
import torch
from transformers import pipeline

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a friendly chatbot who always responds in the style of a pirate",
}


def process_instance(dataset_item: DatasetItem):
    return [SYSTEM_PROMPT, {"role": "user", "content": dataset_item.prompt}]


if __name__ == "__main__":
    print("Loading dataset")
    dataset_iterator = DatasetIterator(
        question_csv_path=Path("data/prompt_selection.csv"),
        jailbreak_csv=Path("data/jailbreak-prompts.csv"),
        use_jailbreak_prompt=False,
        use_blank_image=False,
    )
    print("Loading model")
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Running model")
    outputs = []
    for idx in tqdm(
        range(len(dataset_iterator)), desc="Running model", total=len(dataset_iterator)
    ):
        dataset_item = dataset_iterator[idx]
        messages = [SYSTEM_PROMPT, {"role": "user", "content": dataset_item.prompt}]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = pipe(
            prompt,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1,
            repetition_penalty=1,
        )[0]["generated_text"]
        assert isinstance(answer, str)

        outputs.append((dataset_item.to_key(), answer))

    print("Saving outputs")
    blank_prefix = "blank_" if dataset_iterator.use_blank_image else ""
    output_file = Path(
        f"predictions/{blank_prefix}zephyr7bbeta_jailbreak{dataset_iterator.use_jailbreak_prompt}.json"
    )
    output_file.write_text(json.dumps(dict(outputs), indent=4))
