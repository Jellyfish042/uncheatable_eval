import os
import json


def save_json(my_data, file_name):
    if not os.path.exists('data'):
        os.makedirs('data')

    file_name = file_name.replace('.json', '') + '.json'
    path = os.path.join('data', file_name)

    with open(path, 'w') as f:
        json.dump(my_data, f, ensure_ascii=True, indent=4)
