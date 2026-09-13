from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from models import Pupil
from settings import Settings
from math_server.models import MathPrediction
from math_server.calculator import (
    ParamsInitial,
    ParamsRaw,
    build_params_initial,
    build_params_raw,
    calc_eysenck,
    calc_belbin,
    calc_bennet,
    BELBIN_COLS,
    apply_eysenck_threshold,
    apply_belbin_threshold,
    apply_bennet_threshold,
    calc_recommendation_complex,    
)
from math_server.params import (
    load_params_initial,
    save_params_initial,
    load_params_raw,
    save_params_raw,
)
from math_server.specialist import load_specialists

logger = logging.getLogger(__name__)


def get_math_params(settings: Settings) -> tuple[ParamsInitial, ParamsRaw]:
    """Загрузить или пересчитать параметры для математической модели."""
    logger.info("get_math_params: Starting")
    #В идеале тогда тут нужно запрашивать ВСЕ файлы из определенной выбранной папки
    #Где будут храниться все вычесленные значения
    params_path = settings.cluster_dir / "math_params.json"
    raw_path = settings.cluster_dir / "math_params_raw.json"
    logger.info("get_math_params: params_path = %s", params_path)
    logger.info("get_math_params: raw_path = %s", raw_path)
    #Загрузка параметров, но  сейчас считает для одного, создается 1 файл (без хардкод пути)
    #сделать params_path в зависимости от количества профессий у specialists
    if params_path.exists() and raw_path.exists():
        logger.info("get_math_params: params files exist, trying to load")
        try:
            #Тут загружаем тогда все файлы по профессиям а не один (сейчас он захардкожен в settings)
            #Подумать в чем разница между методами загружающими параметры
            params = load_params_initial(params_path)
            raw = load_params_raw(raw_path)
            logger.info("get_math_params: Successfully loaded params from files")
            return params, raw
        except Exception as e:
            logger.warning("get_math_params: Failed to load math params, recalculating: %s", e)
    #Тут если файлов нету загружаем специалистов, сейчас из settings берется профессия это бред
    #Извлекаем профессии из самих specialists, делаем словарь или нужную структуру
    #Затем под каждую профессию создаем файл с подсчитанными настройками
    logger.info("get_math_params: Recalculating math parameters from specialists...")
    specialists = load_specialists(settings)
    logger.info("get_math_params: Loaded %d specialists", len(specialists))
    #math_profession_filter это то самое зло захардкоженная профессия избавиться от этого
    #На данном этапе после того как специалисты загружены подумать
    #как после их обработки создать файлы по очереди для каждой профессии
    profession_filter = settings.math_profession_filter
    logger.info("get_math_params: profession_filter = %s", profession_filter)

    specialist_dicts = []
    for sp in specialists:
        row = {
            'specialist_id': sp.get('specialist_id', 0),
            'profession': sp.get('profession', ''),
            'extrav_introver_score': sp.get('extrav_introver_score', 0.0),
            'neirotizm_score': sp.get('neirotizm_score', 0.0),
            'engineering_thinking_level': sp.get('engineering_thinking_level', 0.0)
        }
        for col in BELBIN_COLS:
            row[col] = sp.get(col, 0.0)
        specialist_dicts.append(row)

    logger.info("get_math_params: Created %d rows", len(specialist_dicts))
    df = pd.DataFrame(specialist_dicts)
    logger.info("get_math_params: DataFrame shape = %s", df.shape)

    if profession_filter:
        df = df[df['profession'].str.lower() == profession_filter.lower()]
        logger.info("get_math_params: After filter, DataFrame shape = %s", df.shape)

    if len(df) == 0:
        logger.error("get_math_params: No specialists found for profession: %s", profession_filter)
        raise ValueError(f"No specialists found for profession: {profession_filter}")
    #Тут уже остаются специалисты только с одной профессией, но должны быть все профессии(считаем для каждой профессии)
    #Тоесть мы должны сделать это извлечение для всех и вернуть уже не params, raw для одной профессии
    #А для всех и потом работать уже с каждолй (по факту после возвращанеие всех, переб
    specialists_list = df.to_dict('records')
    params = build_params_initial(specialists_list)
    raw = build_params_raw(specialists_list, params)

    save_params_initial(params, params_path)
    save_params_raw(raw, raw_path)
    logger.info("get_math_params: Math parameters saved to %s and %s", params_path, raw_path)

    return params, raw


def predict_math(pupil: Pupil, settings: Settings) -> MathPrediction:
    
    logger.info("math prediction started pupilId=%s", pupil.pupil_id)
    
    try:
        #Тоесть тут мы возвращаем уже не params_ params_raw, а структуру для ВСЕХ вычлененных профессий
        params, params_raw = get_math_params(settings)
        row = {'age': pupil.age if pupil.age is not None else 14}
        #Подумать почему делает одно и тоже при разных условиях, обьединить ?
        #Предобработка pupil
        for test in pupil.psych_tests.values():
            for param in test.psych_params:
                if param.name in ('extrav_introver_score', 'neirotizm_score', 'engineering_thinking_level'):
                    row[param.name] = param.value or 0.0
                elif param.name in BELBIN_COLS:
                    row[param.name] = param.value or 0.0
        #а это уже должно вычисляться для каждой профессии
        eysenck_norm = apply_eysenck_threshold(calc_eysenck(row, params), params_raw)
        
        belbin_norm = apply_belbin_threshold(calc_belbin(row, params), params_raw)
        
        bennet_norm = apply_bennet_threshold(row, params, calc_bennet(row, params))

        total_score = eysenck_norm + belbin_norm + bennet_norm

        weights = params.weights
        additive_utility = (
            eysenck_norm * weights['eysenck'] +
            belbin_norm * weights['belbin'] +
            bennet_norm * weights['bennet']
        )

        weighted_product = (
            eysenck_norm ** weights['eysenck'] +
            belbin_norm ** weights['belbin'] +
            bennet_norm ** weights['bennet']
        )

        norm_utility = (1 + additive_utility) / (1 + params_raw.max_utility)
        norm_product = (1 + weighted_product) / (1 + params_raw.max_product)
        final_index = (norm_utility + norm_product) / 2
        percent = min(final_index * 100, 100)

        recommendation = "рекомендуем" if total_score > 75 else "не рекомендуем"
        recommendation_complex = calc_recommendation_complex(row, recommendation)
        #должны вернуть не  1 result для захардкоженной профессии, а список results для всех
        #Вычлененных профессий
        result = MathPrediction(
            pupilId=pupil.pupil_id,
            percentage=round(percent, 1),
            recommendation=recommendation,
            recommendationComplex=recommendation_complex,
            aizenNorm= round(eysenck_norm, 2),
            belbinNorm= round(belbin_norm, 2),
            bennetNorm= round(bennet_norm, 2),
            finalScore= round(total_score, 2),
            utility= round(additive_utility, 4) #TODO profession, ,multiplicative?
        )

        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise