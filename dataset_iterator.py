import os
from typing import Any, Optional
from dataclasses import dataclass
from PIL import Image

import torch
import pandas as pd
from pydantic import BaseModel, model_validator


class Question(BaseModel):
    category: str
    index: int
    prompt: str

class Jailbreak(BaseModel):
    id: int
    text: str

    @model_validator(mode="before")
    @classmethod
    def validate_instance(cls, field_values: dict[str, Any]) -> dict[str, Any]:
        """Verify that the text field is formattable with the question/prompt."""
        text = field_values["text"]
        if not "[INSERT PROMPT HERE]" in text:
            raise ValueError(f"Jailbreak text does not contain '[INSERT PROMPT HERE]': {text}")
        return field_values


class Instance(BaseModel):
    question: Question
    jailbreak: Optional[Jailbreak] = None
    image_path: str


@dataclass
class DatasetItem:
    prompt: str
    image: Image
    category: str
    index: int
    jailbreak_id: Optional[int] = None


def read_csv(path: str):
    return pd.read_csv(path)
    #return pd.read_csv(path, header=None).values.tolist()[1:]


class DatasetIterator(torch.utils.data.Dataset):
    def __init__(self,
        question_csv_path: str,
        jailbreak_csv: str,
        images_folder_path: str = "data/images",
        use_jailbreak_prompt: bool = False,
        use_blank_image: bool = False
    ):

        self.use_jailbreak_prompt = use_jailbreak_prompt
        self.use_blank_image = use_blank_image
        self.images_folder_path = images_folder_path

        self.dataset = self._build_dataset(
            question_csv_path, jailbreak_csv,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Instance:
        instance = self.dataset[idx]
        if instance.jailbreak is not None:
            prompt = instance.jailbreak.text.replace("[INSERT PROMPT HERE]", instance.question.prompt)
        else:
            prompt = instance.question.prompt

        return DatasetItem(
            prompt=prompt,
            image=Image.open(instance.image_path),
            category=instance.question.category,
            index=instance.question.index,
            jailbreak_id=instance.jailbreak.id if instance.jailbreak is not None else None
        )

    def _build_dataset(self,
        question_csv_path: str,
        jailbreak_csv: str,
    ) -> list[Instance]:
        questions = read_csv(question_csv_path)
        jailbreak = read_csv(jailbreak_csv) if self.use_jailbreak_prompt else None

        dataset = []
        for _, raw_question in questions.iterrows():
            question = Question(**raw_question)
            image_path = self._get_image_path(question)
            if jailbreak is not None:
                for _, raw_jailbreak in jailbreak.iterrows():
                    jailbreak_model = Jailbreak.model_validate(raw_jailbreak.to_dict())
                    dataset.append(Instance(question=question, jailbreak=jailbreak_model, image_path=image_path))
            else:
                dataset.append(Instance(question=question, jailbreak=None, image_path=image_path))
        return dataset

    def _get_image_path(self, question: Question):
        if self.use_blank_image:
            image_path = os.path.join(self.images_folder_path, "blank.jpg")
        else:
            image_path = os.path.join(self.images_folder_path, f"{question.category}_{question.index}.jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path



# dataset = DatasetIterator(
#     question_csv_path="data/prompt_selection.csv",
#     jailbreak_csv="data/jailbreak-prompts.csv",
#     images_folder_path="data/images",
#     use_jailbreak_prompt=True,
#     use_blank_image=False
# )

# for instance in dataset:
#     breakpoint()
