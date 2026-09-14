from __future__ import annotations

import json
import pickle
from pathlib import Path

from math_server.calculator import ParamsInitial, ParamsRaw


def load_params_initial(path: Path) -> ParamsInitial:
    """Загрузить параметры из JSON."""
    data = json.loads(path.read_text(encoding='utf-8'))
    return ParamsInitial(
        extroversion=data['extroversion'],
        neuroticism=data['neuroticism'],
        bennet=data['bennet'],
        belbin=data['belbin'],
        weights=data['weights']
    )


def save_params_initial(params: ParamsInitial, path: Path) -> None:
    """Сохранить параметры в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(params.__dict__, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )


def load_params_raw(path: Path) -> ParamsRaw:
    """Загрузить сырые параметры из JSON."""
    data = json.loads(path.read_text(encoding='utf-8'))
    return ParamsRaw(
        eysenck=data['eysenck'],
        bennet=data['bennet'],
        belbin=data['belbin'],
        weights=data['weights'],
        max_utility=data['max_utility'],
        max_product=data['max_product'],
    )


def save_params_raw(params: ParamsRaw, path: Path) -> None:
    """Сохранить сырые параметры в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(params.__dict__, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

def profession_slug(profession: str) -> str:
    """Безопасное имя файла из названия профессии."""
    import re
    slug = re.sub(r"[^\w\-]+", "_", profession.strip().lower(), flags=re.UNICODE)
    return slug.strip("_") or "unknown"


def params_paths_for(folder: Path, profession: str) -> tuple[Path, Path]:
    slug = profession_slug(profession)
    return folder / f"{slug}.json", folder / f"{slug}_raw.json"