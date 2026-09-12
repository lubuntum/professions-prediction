"""Load specialists for math prediction."""

from __future__ import annotations

import logging
from typing import Any

import requests
from urllib.parse import urljoin

from settings import Settings

logger = logging.getLogger(__name__)


def load_specialists(settings: Settings) -> list[dict[str, Any]]:
    """Load Specialist data from backend and convert to dict format."""
    logger.info("load_specialists: Starting")
    
    if not settings.backend_url or not settings.backend_email or not settings.backend_password:
        logger.error("load_specialists: Missing credentials")
        raise RuntimeError(
            "BACKEND_BASE_URL, BACKEND_SERVICE_EMAIL and BACKEND_SERVICE_PASSWORD are required"
        )

    timeout = (settings.backend_connect_timeout, settings.backend_read_timeout)
    logger.info("load_specialists: Logging in to %s", settings.backend_url)
    
    try:
        login = requests.post(
            urljoin(settings.backend_url.rstrip("/") + "/", "api/auth/login"),
            json={"email": settings.backend_email, "password": settings.backend_password},
            timeout=timeout,
        )
        login.raise_for_status()
        token = login.text.strip()
        logger.info("load_specialists: Login successful")
    except Exception as e:
        logger.exception("load_specialists: Login failed")
        raise
    
    if not token:
        logger.error("load_specialists: Empty token")
        raise RuntimeError("Backend returned an empty service token")

    try:
        response = requests.get(
            urljoin(settings.backend_url.rstrip("/") + "/", "api/specialists/reference-data"),
            headers={"Authorization": token},
            timeout=timeout,
        )
        response.raise_for_status()
        specialists = response.json()
        logger.info("load_specialists: Got %d specialists", len(specialists) if specialists else 0)
    except Exception as e:
        logger.exception("load_specialists: Failed to get specialists")
        raise
    
    if not isinstance(specialists, list):
        logger.error("load_specialists: Response is not a list")
        raise RuntimeError("Backend Specialist response is not a list")
    
    # Преобразуем в словари
    result = []

    for sp in specialists:
        try:
            row = {
                'specialist_id': sp.get('specialistId', 0),
                'profession': sp.get('profession', ''),
                'extrav_introver_score': 0.0,
                'neirotizm_score': 0.0,
                'engineering_thinking_level': 0.0,
                'company_worker': 0.0,
                'chairman': 0.0,
                'shaper': 0.0,
                'plant': 0.0,
                'resource_investigator': 0.0,
                'monitor_evaluation': 0.0,
                'team_worker': 0.0,
                'completer_finisher': 0.0
            }

            # Получаем psychTests как словарь
            psych_tests = sp.get('psychTests', {})
            has_data = False
            # Итерируемся по значениям словаря (данным тестов)
            for test_data in psych_tests.values():
                # Проверяем, что test_data - это словарь и содержит psychParams
                if isinstance(test_data, dict) and 'psychParams' in test_data:
                    for param in test_data.get('psychParams', []):
                        name = param.get('name')
                        value = param.get('param')
                        if name in row and value is not None:
                            row[name] = float(value)
                            if float(value) != 0:  # Если есть ненулевое значение
                                has_data = True
            if not has_data:
                continue

            result.append(row)
        except Exception as e:
            print(f"Ошибка при обработке specialist_id {sp.get('specialistId', 'unknown')}: {e}")
            continue
    
    logger.info("load_specialists: Finished, returning %d rows", len(result))
    return result