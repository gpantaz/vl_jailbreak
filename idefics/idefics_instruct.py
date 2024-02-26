import argparse
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from transformers.generation import GenerationConfig

from dataset_iterator import DatasetIterator
from tqdm import tqdm


def main(args):
    dataset = DatasetIterator(
        question_csv_path=args.question_csv_path,
        jailbreak_csv=args.jailbreak_csv,
        images_folder_path=args.images_folder_path,
        use_jailbreak_prompt=args.use_jailbreak_prompt,
        use_blank_image=args.use_blank_image,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16
    ).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint)

    output_json = args.output_json
    for example in tqdm(dataset, total=len(dataset), desc="Running model"):
        # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
        prompts = [
            [
                f"User: {example.prompt}<end_of_utterance>",
                example.image,
                "\nAssistant:",
            ],
        ]

        # --batched mode
        inputs = processor(
            prompts, add_end_of_utterance_token=False, return_tensors="pt"
        ).to(device)
        # --single sample mode
        # inputs = processor(prompts[0], return_tensors="pt").to(device)

        # Generation args
        exit_condition = processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids
        bad_words_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

        generation_config = GenerationConfig(
            max_length=1024,
            top_k=1,
            top_p=1,
            temperature=1,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=1024,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
        )

        generated_ids = model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            generation_config=generation_config,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for i, t in enumerate(generated_text):
            print(f"{i}:\n{t}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question-csv-path",
        type=str,
        default="data/prompt_selection.csv",
        help="Path to the question csv file.",
    )
    parser.add_argument(
        "--jailbreak-csv",
        type=str,
        default="data/jailbreak-prompts.csv",
        help="Path to the jailbreak csv file.",
    )
    parser.add_argument(
        "--images-folder-path",
        type=str,
        default="data/images",
        help="Path to the images folder.",
    )
    parser.add_argument(
        "--use-jailbreak-prompt",
        action="store_true",
        help="Whether use jailbreak prompt.",
    )
    parser.add_argument(
        "--use-blank-image",
        action="store_true",
        help="Whether use blank image.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="predictions/interlm_xcomposer.json",
        help="Path to the output json file.",
    )
    args = parser.parse_args()

    main(args)
