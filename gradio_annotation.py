import argparse
import os
import json
from typing import Literal

import gradio as gr
from loguru import logger
from PIL import Image


class GradioDemo:
    """Gradio Demo for mamba."""

    def __init__(
        self,
        predictions_folder: str,
        annotation_id: str,
        output_dir: str,
    ) -> None:
        self.predictions_folder = predictions_folder
        self.annotation_id = annotation_id
        self.output_dir = output_dir
        examples = []
        for prediction_file in os.listdir(predictions_folder):
            with open(os.path.join(predictions_folder, prediction_file), "r") as f:
                predictions = json.load(f)
                for prediction_id, prediction in predictions.items():
                    examples.append(
                        {
                            "example_id": f"{prediction_file}_{prediction_id}",
                            "prediction": prediction,
                        }
                    )
        logger.info(f"Found {len(examples)} examples")
        self.examples = examples

    def _load_annotation(self) -> dict[str, bool]:
        data = {}

        annotation_file = os.path.join(self.output_dir, self.annotation_id)
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
        return data

    def _save_annotation(self, annotation: dict[str, bool]) -> None:
        annotation_file = os.path.join(self.output_dir, self.annotation_id)
        data = {}
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
            data.update(annotation)

        with open(annotation_file, "w") as f:
            json.dump(data, f, indent=4)

    def get_next_example_index(self, example_index: int) -> int:
        return min(example_index + 1, len(self.examples) - 1)

    def get_previous_example_index(self, example_index: int) -> int:
        return max(example_index - 1, 0)

    def get_next_unlabeled_example_index(self, example_index: int) -> int:
        data = self._load_annotation()
        for index, example in enumerate(self.examples):
            if example["example_id"] not in data:
                return index
        # If all examples are annotated, return the last one.
        return example_index

    def get_example(
        self, example_index: int, mode: Literal["next", "previous", "unlabeled"]
    ) -> str:
        if mode == "next":
            new_example_index = self.get_next_example_index(example_index)
        elif mode == "previous":
            new_example_index = self.get_previous_example_index(example_index)
        else:
            new_example_index = self.get_next_unlabeled_example_index(example_index)

        # TODO: Add question here
        return (
            new_example_index,
            self.examples[new_example_index]["prediction"],
            self.examples[new_example_index]["prediction"],
        )


def main(args: argparse.Namespace) -> None:
    """Main."""
    gradio_demo = GradioDemo(
        predictions_folder=args.predictions_folder,
        annotation_id=args.annotation_id,
        output_dir=args.output_dir,
    )

    with gr.Blocks() as block:
        example_index = gr.State(gradio_demo.get_next_unlabeled_example_index(0))

        with gr.Row():
            question_prompt = gr.Textbox(
                value=gradio_demo.get_example(example_index.value),
                interactive=False,
                label="Question Prompt",
            )

        with gr.Row():
            prediction = gr.Textbox(
                value=gradio_demo.get_example(example_index.value),
                interactive=False,
                label="Prediction",
            )

        with gr.Row():
            with gr.Column():
                previous_button = gr.Button("Previous", variant="secondary")
                next_button = gr.Button("Next", variant="primary")
                next_unlabeled_button = gr.Button("Next Unlabeled", variant="primary")

        previous_button.click(
            fn=gradio_demo.get_example,
            inputs=[example_index, "previous"],
            outputs=[example_index, question_prompt, prediction],
        )

        next_button.click(
            fn=gradio_demo.get_example,
            inputs=[example_index, "next"],
            outputs=[example_index, question_prompt, prediction],
        )

        next_unlabeled_button.click(
            fn=gradio_demo.get_example,
            inputs=[example_index, "unlabeled"],
            outputs=[example_index, question_prompt, prediction],
        )

        block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-folder",
        default="predictions",
        help="Path to predictions folder contain json files",
    )

    parser.add_argument(
        "--annotation-id",
        help="Annotation id",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="annotations",
        help="Output directory",
    )

    args = parser.parse_args()
    main(args)
