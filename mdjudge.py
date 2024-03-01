import json
import os

from pydantic import BaseModel
from saladbench.evaluator import MDJudgeEvaluator

from dataset_iterator import DatasetIterator


class Sample(BaseModel):
    question: str
    answer: str
    sample_id: str


def get_dataset_iterator(model_predictions_file: str) -> DatasetIterator:
    use_jailbreak_prompt = "jailbreakTrue" in model_predictions_file
    dataset = DatasetIterator(
        question_csv_path="data/prompt_selection.csv",
        jailbreak_csv="data/jailbreak-prompts.csv",
        use_jailbreak_prompt=use_jailbreak_prompt,
        use_blank_image=True,
    )
    return dataset


def prepare_data(model_predictions_file: str) -> list[Sample]:
    dataset = get_dataset_iterator(model_predictions_file)
    model_predictions = json.load(
        open(os.path.join("predictions", model_predictions_file), "r")
    )
    samples = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item.jailbreak_id is not None and item.jailbreak_id == 75:
            continue
        identifier = f"{model_predictions_file}_{item.category}_question{item.index}_jailbreak{item.jailbreak_id}"
        samples.append(
            Sample(
                question=item.prompt,
                answer=model_predictions[identifier],
                sample_id=identifier,
            )
        )
    return samples


def evaluate_samples(data: list[Sample], md_judge: MDJudgeEvaluator) -> None:
    questions = [d.question for d in data]
    answers = [d.answer for d in data]

    eval_labels, eval_clses = md_judge.eval_unsafe_labels(questions, answers)
    print("=" * 20 + "Safety Judgement" + "=" * 20)
    for i in range(len(questions)):
        print("\n===Evaluation===")
        print(["SAFE", "UNSAFE", "Exception"][eval_labels[i]])
        print("===Harm category (maybe empty)===")
        print(eval_clses[i])


def main() -> None:
    md_judge = MDJudgeEvaluator(
        "OpenSafetyLab/MD-Judge-v0.1", use_vllm=False, use_flash_attn=True
    )
    for model_predictions_file in os.listdir("predictions"):
        if "zephyr" in model_predictions_file or "omni" in model_predictions_file:
            continue
        data = prepare_data(model_predictions_file)
        evaluate_samples(data, md_judge)


if __name__ == "__main__":
    main()
