from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from models import Pupil
from settings import Settings
from math_server.models import MathPrediction
from math_server.calculator import (
    ParamsInitial,
    build_params_initial,
    calc_eysenck,
    calc_belbin,
    calc_bennet,
    BELBIN_COLS,
)
from math_server.params import load_params_initial, save_params_initial
from math_server.specialist import load_specialists

logger = logging.getLogger(__name__)


def get_math_params(settings: Settings) -> ParamsInitial:
    print(">>> get_math_params START")
    """Загрузить или пересчитать параметры для математической модели."""
    logger.info("get_math_params: Starting")
    params_path = settings.cluster_dir / "math_params.json"
    logger.info("get_math_params: params_path = %s", params_path)
    
    if params_path.exists():
        logger.info("get_math_params: params file exists, trying to load")
        try:
            result = load_params_initial(params_path)
            logger.info("get_math_params: Successfully loaded params from file")
            return result
        except Exception as e:
            logger.warning("get_math_params: Failed to load math params, recalculating: %s", e)
    
    logger.info("get_math_params: Recalculating math parameters from specialists...")
    specialists = load_specialists(settings)
    logger.info("get_math_params: Loaded %d specialists", len(specialists))
    
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
    
    params = build_params_initial(df.to_dict('records'))
    save_params_initial(params, params_path)
    logger.info("get_math_params: Math parameters saved to %s", params_path)
    
    return params


def predict_math(pupil: Pupil, settings: Settings) -> MathPrediction:
    """Calculate math prediction for pupil."""
    print("=" * 50)
    print("PREDICT_MATH CALLED")
    print("pupil_id:", pupil.pupil_id)
    print("=" * 50)
    
    logger.info("math prediction started pupilId=%s", pupil.pupil_id)
    
    try:
        print("Step 1: Getting math params...")
        params = get_math_params(settings)
        print("Step 1: OK")
        
        print("Step 2: Extracting pupil data...")
        row = {'age': 14}
        
        for test in pupil.psych_tests.values():
            for param in test.psych_params:
                if param.name in ('extrav_introver_score', 'neirotizm_score', 'engineering_thinking_level'):
                    row[param.name] = param.value or 0.0
                elif param.name in BELBIN_COLS:
                    row[param.name] = param.value or 0.0
        
        print("Step 2: Row =", row)
        
        print("Step 3: Calculating norms...")
        eysenck_norm = calc_eysenck(row, params)
        print("Step 3a: Eysenck =", eysenck_norm)
        
        belbin_norm = calc_belbin(row, params)
        print("Step 3b: Belbin =", belbin_norm)
        
        bennet_norm = calc_bennet(row, params)
        print("Step 3c: Bennet =", bennet_norm)
        
        print("Step 4: Total score...")
        total_score = eysenck_norm + belbin_norm + bennet_norm
        print("Step 4: Total =", total_score)
        
        print("Step 5: Additive utility...")
        weights = params.weights
        additive_utility = (
            eysenck_norm * weights['eysenck'] +
            belbin_norm * weights['belbin'] +
            bennet_norm * weights['bennet']
        )
        print("Step 5: Utility =", additive_utility)
        
        print("Step 6: Percent...")
        percent = min(additive_utility / 0.7, 100)
        print("Step 6: Percent =", percent)
        
        print("Step 7: Recommendation...")
        recommendation = "рекомендуем" if total_score > 69 else "не рекомендуем"
        print("Step 7: Recommendation =", recommendation)
        
        print("Step 8: Creating result...")
        result = MathPrediction(
            pupil_id=pupil.pupil_id,
            percent=round(percent, 1),
            recommendation=recommendation,
            scores={
                "Айзенк_норм": round(eysenck_norm, 2),
                "Белбин_норм": round(belbin_norm, 2),
                "Беннет_норм": round(bennet_norm, 2),
                "Итоговый_балл": round(total_score, 2),
                "Аддитивная_полезность": round(additive_utility, 4)
            }
        )
        
        print("=" * 50)
        print("PREDICT_MATH SUCCESS")
        print("=" * 50)
        return result
        
    except Exception as e:
        print("!" * 50)
        print("ERROR in predict_math:")
        print(type(e).__name__, str(e))
        import traceback
        traceback.print_exc()
        print("!" * 50)
        raise