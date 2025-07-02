import os
import json
import traceback
import requests
from flask import Flask, render_template, jsonify, request

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
LOCAL_MODEL_NAME = "llama3:8b" 

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

TASK_GENERATION_PROMPT_TEMPLATE = """
Выступайте в роли технического наставника по Python. Ваша задача — создать учебное задание.

Задание должно содержать:
1.  **title**: Название задания на русском языке.
2.  **task**: Четкое описание условия задачи на русском языке.
3.  **buggy_code**: Код функции на Python, который содержит **обязательную** и **неочевидную** логическую ошибку, связанную с темой "{theme}". Код не должен содержать синтаксических ошибок.

Ваш ответ должен быть представлен СТРОГО в формате JSON-строки без каких-либо вводных слов или комментариев.
"""

SOLUTION_VERIFICATION_PROMPT_TEMPLATE = """
Проанализируйте предоставленный код на соответствие техническому заданию.

**Техническое задание:**
{task_description}

**Код для проверки:**
```python
{user_code}
```

**Ваша цель:**
Определить, соответствует ли код заданию.

Ваш ответ должен быть представлен СТРОГО в формате JSON-строки со следующими ключами:
1.  **is_correct**: булево значение (true, если код полностью и корректно решает задачу; в противном случае — false).
2.  **explanation**: Суть ошибки или подтверждение корректности решения на русском языке.
"""

def query_local_model(prompt):
    """
    Отправляет запрос к локальной модели через API Ollama.
    """
    try:
        payload = {
            "model": LOCAL_MODEL_NAME,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        response_json_string = response.json().get('response', '{}')
        return json.loads(response_json_string)
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Не удалось подключиться к Ollama. Убедитесь, что Ollama запущен.")
    except Exception as e:
        raise Exception(f"Ошибка при работе с локальной моделью: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-task')
def get_task():
    """
    Генерирует задачу с учетом выбранной темы и выполняет двойную проверку.
    """
    selected_theme = request.args.get('theme', 'общие алгоритмы')
    print(f"Запрошена тема: {selected_theme}")

    for attempt in range(5):
        try:
            print(f"Попытка №{attempt + 1}: Генерация задачи по теме '{selected_theme}'...")
            generation_prompt = TASK_GENERATION_PROMPT_TEMPLATE.format(theme=selected_theme)
            task_data = query_local_model(generation_prompt)

            if not all(k in task_data for k in ['task', 'buggy_code', 'title']):
                print("Сгенерированные данные неполные, повторная попытка...")
                continue

            print("Двойная проверка: Анализ сгенерированного кода...")
            verification_prompt = SOLUTION_VERIFICATION_PROMPT_TEMPLATE.format(
                task_description=task_data['task'],
                user_code=task_data['buggy_code']
            )
            review_data = query_local_model(verification_prompt)

            if not review_data.get('is_correct'):
                print("Проверка пройдена: Код действительно содержит ошибку.")
                return jsonify(task_data)
            else:
                print("Двойная проверка не пройдена: сгенерированный код не содержит ошибки. Повторная генерация...")
        except Exception as e:
            print(f"Ошибка на попытке №{attempt + 1}: {e}")
            continue
    
    print("Не удалось сгенерировать корректную задачу после нескольких попыток.")
    return jsonify({"error": "Не удалось сгенерировать качественную задачу. Попробуйте снова или выберите другую тему."}), 500

@app.route('/check-solution-with-llm', methods=['POST'])
def check_solution_with_llm():
    data = request.json
    user_code = data.get('code')
    task_description = data.get('task')

    if not all([user_code, task_description]):
        return jsonify({"error": "Отсутствуют данные для проверки."}), 400
    try:
        verification_prompt = SOLUTION_VERIFICATION_PROMPT_TEMPLATE.format(
            task_description=task_description,
            user_code=user_code
        )
        print("Запрос к локальной модели: проверка решения...")
        review_data = query_local_model(verification_prompt)
        print("Ревью от локальной модели получено.")
        return jsonify(review_data)
    except Exception as e:
        print(f"Ошибка при проверке решения: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
