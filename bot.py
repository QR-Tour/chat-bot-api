from flask import Flask, jsonify, request
import requests
from flask_cors import CORS
from deeppavlov import build_model
from deeppavlov.core.common.file import read_json

app = Flask(__name__)
CORS(app)

model_config_bert = read_json("squad_ru_bert_infer.json")
model_bert = build_model(model_config_bert, download=True)

def is_valid_id(id):
    return isinstance(id, int) or (isinstance(id, str) and id.isdigit())

def get_data_from_api(id):
    if is_valid_id(id):
        url = f"http://127.0.0.1:5000/{id}"
        response = requests.get(url)
        return response.json()
    else:
        return jsonify({'status': '1', 'message': 'Некорректный QR-код'})


@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get('data')
    id = data.get('id')

    data_from_api = get_data_from_api(id)

    if not ('status' in data_from_api):
        answer = model_bert([data_from_api['desc']], [question])

        if answer[0][0].strip() and answer[2][0] > 0.9998:
            return jsonify({"data": answer[0][0]})
        elif answer[2][0] < 1 and answer[0][0].strip():
            return jsonify({"data": "Я не могу дать точный ответ на вопрос"})
        else:
            return jsonify({"data": "Вопрос не по тексту"})
    else:
        return data_from_api

@app.route('/', methods=['POST'])
def get_desc():
    data = request.get_json()
    id = data.get('id')

    data_from_api = get_data_from_api(id)

    if not ('status' in data_from_api):
        return jsonify({"data": data_from_api['desc']})
    else:
        return data_from_api

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
