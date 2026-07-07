import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

from langgraph_harness.schemas import AttemptRecord, FinalReport


class RunLogger:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, data: Dict[str, Any]) -> str:
        path = self.artifact_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return str(path)

    def write_attempt(self, record: AttemptRecord) -> str:
        path = self.artifact_dir / "attempts.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
            )
        return str(path)

    def write_image(self, iteration: int, content: bytes) -> str:
        path = self.artifact_dir / "round_{:02d}.png".format(iteration)
        path.write_bytes(content)
        return str(path)

    def write_ocr_result(self, iteration: int, data: Dict[str, Any]) -> str:
        return self.write_json("ocr_round_{:02d}.json".format(iteration), data)

    def write_final_report(self, report: FinalReport) -> str:
        return self.write_json("final_report.json", report.model_dump())

    @staticmethod
    def common_tags(attempts: Iterable[Dict[str, Any]]) -> list:
        counter = Counter()
        for attempt in attempts:
            counter.update(attempt.get("failure_tags", []))
        return [name for name, _ in counter.most_common()]
