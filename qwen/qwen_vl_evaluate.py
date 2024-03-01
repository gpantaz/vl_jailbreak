import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import sys
from tqdm import tqdm

sys.path.append("./")
from dataset_iterator import DatasetIterator


def main(args):
    dataset = DatasetIterator(
        question_csv_path=args.question_csv_path,
        jailbreak_csv=args.jailbreak_csv,
        images_folder_path=args.images_folder_path,
        use_jailbreak_prompt=args.use_jailbreak_prompt,
        use_blank_image=args.use_blank_image,
    )

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat", trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True
    ).eval()
    generation_config = GenerationConfig(
        max_length=1024,
        top_k=1,
        top_p=args.top_p,
        temperature=1,
        num_return_sequences=1,
        do_sample=args.do_sample,
        eos_token_id=151643,
        max_new_tokens=1024,
        max_window_size=6144,
        pad_token_id=151643,
        repetition_penalty=1.0,
        chat_format="chatml",
    )

    output_json = args.output_json
    for example in tqdm(dataset, total=len(dataset), desc="Running model"):
        # response, history = model.chat(tokenizer, example.prompt, history=None, generation_config=generation_config)

        query = tokenizer.from_list_format(
            [
                # {'image': example.image},
                {"image": str(example.image_path)},
                {"text": example.prompt},
            ]
        )
        response, history = model.chat(
            tokenizer, query=query, history=None, generation_config=generation_config
        )

        # query = tokenizer.from_list_format(
        #     [{'image': "/users/gmp2000/sharedscratch/vl_jailbreak/data/images/harmful_content_0.jpg"},{'text': "a photo of a"},
        #     ])

        if os.path.exists(output_json):
            data = json.load(open(output_json))
        else:
            data = {}

        data[example.to_key()] = response
        with open(output_json, "w") as f:
            json.dump(data, f, indent=4)


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
        default="vicuna_output.json",
        help="Path to the output json file.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
    )
    parser.add_argument(
        "--top-p",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    main(args)
