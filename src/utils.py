import json
import pytz
from typing import Any
from pathlib import Path
from datetime import datetime


def load_jsonl(file_path: Path | str) -> list[dict[str, Any]]:

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                continue

            item = json.loads(line)
            data.append(item)

    return data


def save_jsonl(file_path: Path | str, data: list[dict[str, Any]]) -> None:

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        f.write("\n")


def load_json(file_path: Path | str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(file_path: Path | str, data: list[dict[str, Any]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_current_datetime(tz="Asia/Tokyo") -> datetime:

    # set timezone
    tz = pytz.timezone(tz)

    # get current UTC datetime
    utc_now = datetime.now()

    # convert to JST
    jst_now = utc_now.astimezone(tz)

    return jst_now
