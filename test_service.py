import requests
import logging

logging.basicConfig(filename='test_service.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

features_store_url = "http://0.0.0.0:8001"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def test(params):
    resp = requests.post(features_store_url + "/recommendations", headers=headers, params=params)
    if resp.status_code == 200:
        result = resp.json()
        logging.info(f"Recommendations for user {params['user_id']}: {result}")
    else:
        result = None
        logging.error(f"Failed to get recommendations for user {params['user_id']}, status code: {resp.status_code}")
    return result


def input(params):
    resp = requests.post(features_store_url + "/put", headers=headers, params=params)
    if resp.status_code == 200:
        result = resp.json()
        logging.info(f"Input action for user {params['user_id']} with item {params['item_id']}: {result}")
    else:
        result = None
        logging.error(f"Failed to input action for user {params['user_id']}, status code: {resp.status_code}")

    return result


logging.info('Starting tests...')

print('New user')
params = {"user_id": 17245}
print(test(params))  # Ожидаем что на результат будет 100 топовых треков из файла top_popular

print('User with personal recommendations')
params = {"user_id": 982299}
print(test(params))  # Ожидаем что на результат будет 10 персональных треков из файла personal

print('Some action for user 982299')
params = {"user_id": 982299, "item_id": 56052565}
print(input(params))

print('New recommendations personal with online')
params = {"user_id": 982299}
print(test(params))  # Ожидаем что на результат будет включать 10 персональных треков из файла personal + новые онлайн

logging.info('Tests completed.')