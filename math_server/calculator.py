from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Any


@dataclass
class ParamsInitial:
    extroversion: dict[str, float]
    neuroticism: dict[str, float]
    bennet: dict[str, float]
    belbin: dict[str, Any]
    weights: dict[str, float]


@dataclass
class ParamsRaw:
    eysenck: dict[str, float]
    bennet: dict[str, float]
    belbin: dict[str, float]
    weights: dict[str, float]


# ===== ВОЗРАСТНАЯ КОРРЕКТИРОВКА =====
def adjust_extroversion(value: float, age: int) -> float:
    """=E4+(12-E4)*(1-C4/21)"""
    return value + (12 - value) * (1 - age / 21)


def adjust_neuroticism(value: float, age: int) -> float:
    """=F4+(10-F4)*(1-C4/21)"""
    return value + (10 - value) * (1 - age / 21)


def adjust_worker_bee(value: float, age: int) -> float:
    """=K4+(15-K4)*(1-C4/19)"""
    return value + (15 - value) * (1 - age / 19)


def adjust_leader(value: float, age: int) -> float:
    """=L4+(14-L4)*(1-C4/24)"""
    return value + (14 - value) * (1 - age / 24)


def adjust_motivator(value: float, age: int) -> float:
    """=M4-(M4-12)*(1-C4/21)"""
    return value - (value - 12) * (1 - age / 21)


def adjust_idea_generator(value: float, age: int) -> float:
    """=N4-(N4-11)*(1-C4/19)"""
    return value - (value - 11) * (1 - age / 19)


def adjust_supplier(value: float, age: int) -> float:
    """=O4+(13-O4)*(1-C4/21)"""
    return value + (13 - value) * (1 - age / 21)


def adjust_analyst(value: float, age: int) -> float:
    """=P4+(14-P4)*(1-C4/19)"""
    return value + (14 - value) * (1 - age / 19)


def adjust_inspirer(value: float, age: int) -> float:
    """=Q4-(Q4-10)*(1-C4/22)"""
    return value - (value - 10) * (1 - age / 22)


def adjust_controller(value: float, age: int) -> float:
    """=R4+(13-R4)*(1-C4/19)"""
    return value + (13 - value) * (1 - age / 19)


def adjust_bennet(value: float, age: int) -> float:
    """=AC4+(52-AC4)*(1-C4/17)"""
    return value + (52 - value) * (1 - age / 17)


BELBIN_COLS = [
    'company_worker', 'chairman', 'shaper', 'plant',
    'resource_investigator', 'monitor_evaluation', 'team_worker', 'completer_finisher'
]

ADJUST_FUNCTIONS = {
    'extrav_introver_score': adjust_extroversion,
    'neirotizm_score': adjust_neuroticism,
    'company_worker': adjust_worker_bee,
    'chairman': adjust_leader,
    'shaper': adjust_motivator,
    'plant': adjust_idea_generator,
    'resource_investigator': adjust_supplier,
    'monitor_evaluation': adjust_analyst,
    'team_worker': adjust_inspirer,
    'completer_finisher': adjust_controller,
    'engineering_thinking_level': adjust_bennet
}


# ===== РАСЧЁТ НОРМИРОВАННЫХ БАЛЛОВ =====
def calc_eysenck(row: dict, params: ParamsInitial) -> float:
    """
    =G4*(1-МИН(G4-$F$20;$F$21-G4)/$F$22) + H4*(1-МИН(H4-$G$20;$G$21-H4)/$G$22)
    Проверка интервала: 10 < val < 31
    """
    age = row.get('age', 0)
    e = adjust_extroversion(row.get('extrav_introver_score', 0), age)
    n = adjust_neuroticism(row.get('neirotizm_score', 0), age)

    extro_min = params.extroversion['min']
    extro_max = params.extroversion['max']
    extro_mean = params.extroversion['mean']

    neuro_min = params.neuroticism['min']
    neuro_max = params.neuroticism['max']
    neuro_mean = params.neuroticism['mean']

    val = e * (1 - min(e - extro_min, extro_max - e) / extro_mean) + \
          n * (1 - min(n - neuro_min, neuro_max - n) / neuro_mean)

    return val if 10 < val < 31 else 0


def calc_belbin(row: dict, params: ParamsInitial) -> float:
    """
    Сумма по 8 ролям: s*(1 - min(s - min, max - s) / mean)
    Проверка интервала: 32 < total < 57
    """
    age = row.get('age', 0)
    total = 0

    for i, col in enumerate(BELBIN_COLS):
        s = ADJUST_FUNCTIONS[col](row.get(col, 0), age)

        min_val = params.belbin['mins'][i]
        max_val = params.belbin['maxs'][i]
        mean_val = params.belbin['means'][i]

        total += s * (1 - min(s - min_val, max_val - s) / mean_val)

    return total if 32 < total < 57 else 0


def calc_bennet(row: dict, params: ParamsInitial) -> float:
    """
    =ЕСЛИ(AD4>25;AD4;0) - проверка скорректированного значения
    =AE4*(1-МИН(AE4-$J$20;$J$21-AE4)/$J$22) - нормировка
    """
    age = row.get('age', 0)
    b_orig = row.get('engineering_thinking_level', 0)

    # Скорректированное значение
    b_adj = adjust_bennet(b_orig, age) if b_orig > 0 else 0

    # Проверка: AD4 > 25
    if b_adj <= 25:
        return 0

    # Нормировка
    min_val = params.bennet['min']
    max_val = params.bennet['max']
    mean_val = params.bennet['mean']

    return b_adj * (1 - min(b_adj - min_val, max_val - b_adj) / mean_val)


# ===== ФОРМИРОВАНИЕ ПАРАМЕТРОВ =====
def build_params_initial(specialists: list[dict]) -> ParamsInitial:
    """Рассчитать параметры на основе данных специалистов."""
    import pandas as pd

    df = pd.DataFrame(specialists)

    return ParamsInitial(
        extroversion={
            'min': float(df['extrav_introver_score'].min()),
            'max': float(df['extrav_introver_score'].max()),
            'mean': float(df['extrav_introver_score'].mean())
        },
        neuroticism={
            'min': float(df['neirotizm_score'].min()),
            'max': float(df['neirotizm_score'].max()),
            'mean': float(df['neirotizm_score'].mean())
        },
        bennet={
            'min': float(df['engineering_thinking_level'].min()),
            'max': 70.0,
            'mean': float(df['engineering_thinking_level'].mean())
        },
        belbin={
            'cols': BELBIN_COLS,
            'mins': [float(df[col].min()) for col in BELBIN_COLS],
            'maxs': [float(df[col].max()) for col in BELBIN_COLS],
            'means': [float(df[col].mean()) for col in BELBIN_COLS]
        },
        weights={
            'eysenck': 0.35,
            'bennet': 0.3,
            'belbin': 0.35
        }
    )