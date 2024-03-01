import argparse
import torch
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import json
from transformers.generation import GenerationConfig

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import sys
from tqdm import tqdm

sys.path.append("./")
from dataset_iterator import DatasetIterator


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    dataset = DatasetIterator(
        question_csv_path=args.question_csv_path,
        jailbreak_csv=args.jailbreak_csv,
        images_folder_path=args.images_folder_path,
        use_jailbreak_prompt=args.use_jailbreak_prompt,
        use_blank_image=args.use_blank_image,
    )

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv_mode = args.conv_mode
    # if "mpt" in model_name.lower():
    #     roles = ("user", "assistant")
    # else:
    #     roles = conv.roles

    # # Similar operation in model_worker.py
    # if type(image_tensor) is list:
    #     image_tensor = [
    #         image.to(model.device, dtype=torch.float16) for image in image_tensor
    #     ]
    # else:
    #     image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    for example in tqdm(dataset, total=len(dataset), desc="Running model"):
        conv = conv_templates[conv_mode].copy()
        image = example.image.convert("RGB")
        image_size = image
        image_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=model.dtype
        )

        # try:
        #     inp = input(f"{roles[0]}: ")
        # except EOFError:
        #     inp = ""
        # if not inp:
        #     print("exit...")
        #     break

        # print(f"{roles[1]}: ", end="")
        inp = example.prompt

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_config = GenerationConfig(
            max_length=1024,
            top_k=1,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=1,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device=model.device),
                images=image_tensor,
                image_sizes=[image_size],
                generation_config=generation_config,
                # do_sample=False if args.temperature > 0 else False,
                # temperature=args.temperature,
                # max_new_tokens=args.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
            )

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        output_json = args.output_json
        if os.path.exists(output_json):
            data = json.load(open(output_json))
        else:
            data = {}

        data[
            f"{example.category}_question{example.index}_jailbreak{example.jailbreak_id}"
        ] = outputs
        with open(output_json, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

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
        type=float,
        default=1,
    )
    args = parser.parse_args()
    main(args)
