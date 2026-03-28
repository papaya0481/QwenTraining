import json
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Test:
    input: str
    output: str
    testtype: str


@dataclass
class TestOutputPredictionProblem:
    question_title: str
    question_content: str
    question_id: str
    contest_id: str
    contest_date: datetime
    difficulty: str
    test: list[Test]
    starter_code: str
    function_name: str
    test_id: int

    def __post_init__(self):
        if isinstance(self.contest_date, str):
            self.contest_date = datetime.fromisoformat(self.contest_date)

        parsed_test = json.loads(self.test) if isinstance(self.test, str) else self.test
        if not isinstance(parsed_test, list):
            raise TypeError(f"test must be a list or JSON list string, got {type(parsed_test).__name__}")

        self.test = [t if isinstance(t, Test) else Test(**t) for t in parsed_test]  # type: ignore[arg-type]

    def insert_output(self, output_list: list[str], pred_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "difficulty": self.difficulty,
            "output_list": output_list,
            "pred_list": pred_list,
            "test_id": self.test_id,
            "function_name": self.function_name,
            "starter_code": self.starter_code,
        }

    def insert_output_evaluation(
        self, output_list: list[str], code_list: list[str], graded_list: list[bool]
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        return output

    def get_evaluation_sample(self) -> dict:
        return {
            "input": self.question_content,
            "output": self.test[0].output,
        }


def load_test_prediction_dataset(
    release_version="release_v1", start_date=None, end_date=None
) -> list[TestOutputPredictionProblem]:
    dataset = load_dataset("livecodebench/test_generation", split="test")  # type: ignore
    dataset = [TestOutputPredictionProblem(**d) for d in dataset]

    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} prediction problems")
    return dataset


if __name__ == "__main__":
    dataset = load_test_prediction_dataset()
