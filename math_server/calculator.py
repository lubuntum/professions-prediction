from __future__ import annotations

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
    max_utility: float
    max_product: float


# ===== ВСПОМОГАТЕЛЬНОЕ =====
def _safe_ratio(numerator: float, denominator: float) -> float:
    """x/0 → 0, 0/0 → 0, иначе обычное деление."""
    return numerator / denominator if denominator else 0.0


# ===== ВОЗРАСТНАЯ КОРРЕКТИРОВКА (только для учеников) =====
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
    'engineering_thinking_level': adjust_bennet,
}


# ===== ПОРОГИ ОТСЕЧЕНИЯ (только для учеников) =====
def apply_eysenck_threshold(val: float, raw: ParamsRaw) -> float:
    """Обрезка по порогам из params_raw (только для учеников)."""
    min_eysenck = raw.eysenck['min']
    max_eysenck = raw.eysenck['max']
    return val if (val > min_eysenck - 1 and val < max_eysenck + 1) else 0.0


def apply_belbin_threshold(total: float, raw: ParamsRaw) -> float:
    """Обрезка по нижней границе из params_raw (только для учеников)."""
    min_belbin = raw.belbin['min']
    return total if total > min_belbin - 1 else 0.0


# ===== РАСЧЁТ ДЛЯ СПЕЦИАЛИСТОВ (без age, без порогов) =====
def calc_specialist_eysenck(row: dict, params: ParamsInitial) -> float:
    """Нормированный балл Айзенка для специалиста: сырые значения, без age."""
    e = row['extrav_introver_score']
    n = row['neirotizm_score']
    return (
        e * (1 - _safe_ratio(
            min(e - params.extroversion['min'], params.extroversion['max'] - e),
            params.extroversion['mean'],
        ))
        + n * (1 - _safe_ratio(
            min(n - params.neuroticism['min'], params.neuroticism['max'] - n),
            params.neuroticism['mean'],
        ))
    )


def calc_specialist_belbin(row: dict, params: ParamsInitial) -> float:
    """Сумма по 8 ролям для специалиста: сырые значения, без age."""
    total = 0.0
    for i, col in enumerate(BELBIN_COLS):
        s = row[col]
        min_val = params.belbin['mins'][i]
        max_val = params.belbin['maxs'][i]
        mean_val = params.belbin['means'][i]
        total += s * (1 - _safe_ratio(min(s - min_val, max_val - s), mean_val))
    return total


def calc_specialist_bennet(row: dict, params: ParamsInitial) -> float:
    """Нормированный балл Беннета для специалиста: сырое значение, без age."""
    b = row['engineering_thinking_level']
    return b * (1 - _safe_ratio(
        min(b - params.bennet['min'], params.bennet['max'] - b),
        params.bennet['mean'],
    ))


# ===== РАСЧЁТ ДЛЯ УЧЕНИКОВ (с age, с порогами) =====
def calc_pupil_eysenck(row: dict, params: ParamsInitial, params_raw: ParamsRaw) -> float:
    """Нормированный балл Айзенка для ученика: с age, с порогами."""
    age = row['age']
    e = adjust_extroversion(row['extrav_introver_score'], age)
    n = adjust_neuroticism(row['neirotizm_score'], age)
    val = (
        e * (1 - _safe_ratio(
            min(e - params.extroversion['min'], params.extroversion['max'] - e),
            params.extroversion['mean'],
        ))
        + n * (1 - _safe_ratio(
            min(n - params.neuroticism['min'], params.neuroticism['max'] - n),
            params.neuroticism['mean'],
        ))
    )
    return apply_eysenck_threshold(val, params_raw)


def calc_pupil_belbin(row: dict, params: ParamsInitial, params_raw: ParamsRaw) -> float:
    """Сумма по 8 ролям для ученика: с age, с порогом по нижней границе."""
    age = row['age']
    total = 0.0
    for i, col in enumerate(BELBIN_COLS):
        s = ADJUST_FUNCTIONS[col](row[col], age)
        min_val = params.belbin['mins'][i]
        max_val = params.belbin['maxs'][i]
        mean_val = params.belbin['means'][i]
        total += s * (1 - _safe_ratio(min(s - min_val, max_val - s), mean_val))
    return apply_belbin_threshold(total, params_raw)


def calc_pupil_bennet(row: dict, params: ParamsInitial, params_raw: ParamsRaw) -> float:
    """
    Нормированный балл Беннета для ученика (по users.py):
      (1) если b_adj вне коридора [min, max] -> 0
      (2) если norm_val <= min -> 0
    """
    age = row['age']
    b_orig = row['engineering_thinking_level']
    b_adj = adjust_bennet(b_orig, age) if b_orig > 0 else 0

    min_val = params.bennet['min']
    max_val = params.bennet['max']
    mean_val = params.bennet['mean']

    if b_adj < min_val or b_adj > max_val:
        return 0.0

    norm_val = b_adj * (1 - _safe_ratio(min(b_adj - min_val, max_val - b_adj), mean_val))

    if norm_val > min_val:
        return norm_val
    return 0.0


def calc_recommendation_complex(row: dict, simple_recommendation: str) -> str:
    """
    Сложная рекомендация (по users.py).

    Если простая = "не рекомендуем" — сложная тоже "не рекомендуем".
    Иначе смотрим на:
      s4  = adjust_worker_bee(company_worker, age)     — скорректированная "Рабочая пчелка"
      ac4 = engineering_thinking_level (сырое!)        — сырой Беннет
      ae4 = adjust_bennet(engineering_thinking_level, age), если > 0; при ae4 < 26 -> 0
    Если min(s4, ac4, ae4) == 0 — "рекомендуем частично", иначе — "рекомендуем".
    """
    if simple_recommendation == "не рекомендуем":
        return "не рекомендуем"

    age = row['age']
    company_worker = row['company_worker']
    eng = row['engineering_thinking_level']

    s4 = adjust_worker_bee(company_worker, age)
    ac4 = eng
    ae4 = adjust_bennet(eng, age) if eng > 0 else 0
    if ae4 < 26:
        ae4 = 0

    if min(s4, ac4, ae4) == 0:
        return "рекомендуем частично"
    return "рекомендуем"


# ===== ФОРМИРОВАНИЕ ПАРАМЕТРОВ =====
def build_params_initial(specialists: list[dict]) -> ParamsInitial:
    """Рассчитать params_initial на основе сырых данных специалистов."""
    import pandas as pd

    df = pd.DataFrame(specialists)

    return ParamsInitial(
        extroversion={
            'min': float(df['extrav_introver_score'].min()),
            'max': float(df['extrav_introver_score'].max()),
            'mean': float(df['extrav_introver_score'].mean()),
        },
        neuroticism={
            'min': float(df['neirotizm_score'].min()),
            'max': float(df['neirotizm_score'].max()),
            'mean': float(df['neirotizm_score'].mean()),
        },
        bennet={
            'min': float(df['engineering_thinking_level'].min()),
            'max': float(df['engineering_thinking_level'].max()),
            'mean': float(df['engineering_thinking_level'].mean()),
        },
        belbin={
            'cols': BELBIN_COLS,
            'mins': [float(df[col].min()) for col in BELBIN_COLS],
            'maxs': [float(df[col].max()) for col in BELBIN_COLS],
            'means': [float(df[col].mean()) for col in BELBIN_COLS],
        },
        weights={
            'eysenck': 0.35,
            'bennet': 0.3,
            'belbin': 0.35,
        },
    )


def build_params_raw(specialists: list[dict], params_initial: ParamsInitial) -> ParamsRaw:
    """Рассчитать params_raw на основе нормированных баллов специалистов (без age)."""
    import pandas as pd

    df = pd.DataFrame(specialists)

    df['eysenck_raw'] = df.apply(lambda r: calc_specialist_eysenck(r, params_initial), axis=1)
    df['belbin_raw'] = df.apply(lambda r: calc_specialist_belbin(r, params_initial), axis=1)
    df['bennet_raw'] = df.apply(lambda r: calc_specialist_bennet(r, params_initial), axis=1)

    w = params_initial.weights

    df['total_score'] = (
        w['eysenck'] * df['eysenck_raw']
        + w['bennet'] * df['bennet_raw']
        + w['belbin'] * df['belbin_raw']
    )
    df['weighted_product'] = (
        df['eysenck_raw'] ** w['eysenck']
        + df['belbin_raw'] ** w['belbin']
        + df['bennet_raw'] ** w['bennet']
    )

    return ParamsRaw(
        eysenck={
            'min': float(df['eysenck_raw'].min()),
            'max': float(df['eysenck_raw'].max()),
            'mean': float(df['eysenck_raw'].mean()),
        },
        belbin={
            'min': float(df['belbin_raw'].min()),
            'max': float(df['belbin_raw'].max()),
            'mean': float(df['belbin_raw'].mean()),
        },
        bennet={
            'min': float(df['bennet_raw'].min()),
            'max': float(df['bennet_raw'].max()),
            'mean': float(df['bennet_raw'].mean()),
        },
        weights=w,
        max_utility=float(df['total_score'].max()),
        max_product=float(df['weighted_product'].max()),
    )


def build_params_for_profession(profession_rows: list[dict]) -> tuple[ParamsInitial, ParamsRaw]:
    """Рассчитать params_initial и params_raw для одной профессии."""
    params = build_params_initial(profession_rows)
    raw = build_params_raw(profession_rows, params)
    return params, raw