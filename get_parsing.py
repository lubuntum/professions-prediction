from datetime import datetime
import requests
import pandas as pd
import login

def parsing(parse_type):
    """
    Данная функция будет возвращать пользователей с сайта
    Pupil - ученики
    Specialist - специалисты 
    """

    # Текущая дата и время, текущая
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")

    # Получение токена
    token = login.start()

    # Запрос
    # Выбор URL в зависимости от типа
    if parse_type == "Specialist":
        url = "https://api.evcg.ru/api/specialists/completed-tests"
    elif parse_type == "Pupil":
        url = "https://api.evcg.ru/api/pupils/completed-tests"
    else:
        # Для "All" пока заглушка!!!!!
        print("Режим 'All' требует отдельной обработки")
        return None
    
    params = {
        "type": parse_type,
        "startDate": "2026-04-01",
        "endDate": formatted_date
    }
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Если не достучался, то None
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        # print(f"Найдено пользователей: {len(data)}")
    except requests.exceptions.ReadTimeout:
        return None

    # список нужных тестовых значений
    necessary = ["extravag_introver_score",
                 "neirotizm_score", "company_worker",
                 "chairman", "shaper", "plant",
                 "resource_investigator", "monitor_evaluation",
                 "team_worker", "completer_finisher",
                 "engineering_thinking_level"]

    # user_list - список где будет храниться значения
    user_list = []

    for user in data:
        # user_dic - словарь каждого человека
        user_dic = {}

        # Id
        user_dic["Id"] = user.get('accountId')

        # Извлекаем данные в зависимости от
        # специалист это
        if parse_type == "Specialist" and 'specialist' in user:
            specialist = user['specialist']
            parts = [specialist.get('surname', ''),
                    specialist.get('name', ''),
                    specialist.get('patronymic', '')]
            # ФИО, если есть
            user_dic["fullName"] = " ".join(filter(None, parts))
            user_dic["profession"] = specialist.get('profession')
        # или школьник
        elif parse_type == "Pupil" and 'pupil' in user:
            pupil = user['pupil']
            parts = [pupil.get('surname', ''),
                    pupil.get('name', ''),
                    pupil.get('patronymic', '')]
            # ФИО, если есть
            user_dic["fullName"] = " ".join(filter(None, parts))
            # день рождения
            user_dic["birthday"] = pupil.get('birthday')
            # гендер
            user_dic["gender"] = pupil.get('gender')

        # Вытаскиваем параметры
        for test in user.get('psychTests', []):
            for param in test.get('psychParams', []):
                # берем только те, что нам нужны для кластер.
                if param.get('name') in necessary:
                    user_dic[param.get('name')] = param.get('param')
        user_list.append(user_dic)

    # в датаферйм
    df_user = pd.DataFrame(user_list)
    # Удаляем строки, где есть хотя бы одно пропущенное значение
    # возможно это нужно перенести в другой файл
    df_user = df_user.dropna()
    # печать и сохранение, возвращение
    # df_user.to_csv("test.csv", index=False, encoding="utf-8")
    return df_user

# parsing("Pupil")
# Pupil - ученики
# Specialist - специалисты
