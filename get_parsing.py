from datetime import datetime
import requests
import pandas as pd


def parsing(parse_type="All"):
    """
    Данная функция будет возвращать пользователей с сайта
    По умолчанию type="All" - все пользователи
    Pupil - ученики
    Specialist - специалисты 
    """

    # Текущая дата и время, текущая
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Токен
    with open('token.txt', 'r', encoding="utf-8") as file:
        token = file.read()

    url = "https://api.evcg.ru/api/psych-tests/completed-tests"
    params = {
        "type": parse_type,
        "startDate": "2026-04-01T00:00:00",
        "endDate": formatted_date
    }
    headers = {
        "Authorization": f"Bearer {token}"
    }

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
        user_dic = dict()
        # Id и имя (его может не быть...)
        user_dic["Id"] = user.get('accountId')
        user_dic["fullName"] = user.get('fullName', None)
        # Вытаскиваем параметры
        for test in user.get('psychTests', []):        
            for param in test.get('psychParams', []):
                if param.get('name') in necessary:
                    user_dic[param.get('name')] = param.get('param')
        user_list.append(user_dic)
    df_user = pd.DataFrame(user_list)
    df_user.to_csv("test.csv", index=False, encoding="utf-8")
    print(df_user)
             


    

parsing("ALL")
