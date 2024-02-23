import argparse
import os
import json

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
        annotations = []
        for prediction_file in os.listdir(predictions_folder):
            with open(os.path.join(predictions_folder, prediction_file), "r") as f:
                predictions = json.load(f)
                for prediction_id, prediction in predictions.items():
                    annotations.append(
                        {f"{prediction_file}_{prediction_id}": prediction}
                    )
        logger.info(f"Found {len(annotations)} annotations")
        self.annotations = annotations


def main(args: argparse.Namespace) -> None:
    """Main."""
    gradio_demo = GradioDemo(
        predictions_folder=args.predictions_folder,
        annotation_id=args.annotation_id,
        output_dir=args.output_dir,
    )

    with gr.Blocks(theme=gr.themes.Monochrome()) as block:
        example_index = gr.State(0)

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
        default="gpantaz",
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
