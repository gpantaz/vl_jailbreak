import argparse
import os
import json
from typing import Literal


import gradio as gr
import pandas as pd
from loguru import logger
from PIL import Image


class GradioDemo:
    """Gradio Demo for mamba."""

    def __init__(
        self,
        questions_csv_path: str,
        predictions_folder: str,
        annotation_id: str,
        output_dir: str,
    ) -> None:
        self.annotation_id = annotation_id
        self.output_dir = output_dir

        self.questions = {}
        questions_metadata = pd.read_csv(
            questions_csv_path, header=None
        ).values.tolist()[1:]
        for question_metadata in questions_metadata:
            question_id = f"{question_metadata[0]}_{question_metadata[1]}"
            self.questions[question_id] = question_metadata[2]

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

        annotation_file = os.path.join(self.output_dir, f"{self.annotation_id}.json")
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
        return data

    def _save_annotation(self, annotation: dict[str, bool]) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        data = {}
        annotation_file = os.path.join(self.output_dir, f"{self.annotation_id}.json")
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)

        data.update(annotation)
        with open(annotation_file, "w") as f:
            json.dump(data, f, indent=4)

    def _get_question_id_from_example_id(self, example_id: str) -> str:
        # Example_id = 'vicuna13_jailbreakFalse.json_harmful_content_question0_jailbreakNone
        parts = example_id.split("_")
        return f"{parts[2]}_{parts[3]}_{parts[4].split('question')[1]}"

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
    ) -> tuple[int, str, str, bool, int]:
        if mode == "next":
            new_example_index = self.get_next_example_index(example_index)
        elif mode == "previous":
            new_example_index = self.get_previous_example_index(example_index)
        else:
            new_example_index = self.get_next_unlabeled_example_index(example_index)

        # TODO: Add question here
        new_example = self.examples[new_example_index]
        return (
            new_example_index,
            self.questions[
                self._get_question_id_from_example_id(new_example["example_id"])
            ],
            new_example["prediction"],
            self.is_annotated(new_example_index),
            self.get_total_annotations_str(),
        )

    def get_next_example(self, example_index: int) -> str:
        return self.get_example(example_index, "next")

    def get_previous_example(self, example_index: int) -> str:
        return self.get_example(example_index, "previous")

    def get_unlabeled_example(self, example_index: int) -> str:
        return self.get_example(example_index, "unlabeled")

    def annotate_not_toxic(self, example_index: int) -> None:
        self._save_annotation({self.examples[example_index]["example_id"]: False})
        return self.get_example(example_index, "next")

    def annotate_toxic(self, example_index: int) -> None:
        self._save_annotation({self.examples[example_index]["example_id"]: True})
        return self.get_example(example_index, "next")

    def is_annotated(self, example_index: int) -> bool:
        data = self._load_annotation()
        return self.examples[example_index]["example_id"] in data

    def get_total_annotations_str(self) -> int:
        data = self._load_annotation()
        return f"{len(data)} / {len(self.examples)}"


def main(args: argparse.Namespace) -> None:
    """Main."""
    gradio_demo = GradioDemo(
        questions_csv_path=args.questions_csv_path,
        predictions_folder=args.predictions_folder,
        annotation_id=args.annotation_id,
        output_dir=args.output_dir,
    )

    with gr.Blocks() as block:
        start_index, start_question, start_prediction, annotated, total_ann = (
            gradio_demo.get_unlabeled_example(-1)
        )
        example_index = gr.State(start_index)

        with gr.Row():
            with gr.Column():
                annotated_example = gr.Textbox(
                    value=annotated,
                    interactive=False,
                    label="Annotated Example",
                )
            with gr.Column():
                total_annotations = gr.Textbox(
                    value=total_ann,
                    interactive=False,
                    label="Total Annotated Examples",
                )

        with gr.Row():
            question_prompt = gr.Textbox(
                value=start_question,
                interactive=False,
                label="Question Prompt",
            )

        with gr.Row():
            prediction = gr.Textbox(
                value=start_prediction,
                interactive=False,
                label="Prediction",
            )

        with gr.Row():
            with gr.Column():
                not_toxic_button = gr.Button("Not Toxic", variant="primary")
            with gr.Column():
                toxic_response_button = gr.Button("Toxic", variant="primary")

        with gr.Row():
            with gr.Column():
                previous_button = gr.Button("Previous", variant="secondary")
            with gr.Column():
                next_button = gr.Button("Next", variant="secondary")
            with gr.Column():
                next_unlabeled_button = gr.Button("Next Unlabeled", variant="secondary")

        not_toxic_button.click(
            fn=gradio_demo.annotate_not_toxic,
            inputs=[example_index],
            outputs=[
                example_index,
                question_prompt,
                prediction,
                annotated_example,
                total_annotations,
            ],
        )

        toxic_response_button.click(
            fn=gradio_demo.annotate_toxic,
            inputs=[example_index],
            outputs=[
                example_index,
                question_prompt,
                prediction,
                annotated_example,
                total_annotations,
            ],
        )

        previous_button.click(
            fn=gradio_demo.get_previous_example,
            inputs=[example_index],
            outputs=[
                example_index,
                question_prompt,
                prediction,
                annotated_example,
                total_annotations,
            ],
        )

        next_button.click(
            fn=gradio_demo.get_next_example,
            inputs=[example_index],
            outputs=[
                example_index,
                question_prompt,
                prediction,
                annotated_example,
                total_annotations,
            ],
        )

        next_unlabeled_button.click(
            fn=gradio_demo.get_unlabeled_example,
            inputs=[example_index],
            outputs=[
                example_index,
                question_prompt,
                prediction,
                annotated_example,
                total_annotations,
            ],
        )

        block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-csv-path",
        default="data/prompt_selection.csv",
        help="Path to question csv file",
    )
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

    parser.add_argument("--share", action="store_true", help="Share the app")

    args = parser.parse_args()
    main(args)
