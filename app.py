import os
import json
from flask import Flask, render_template, jsonify, request
from openai import OpenAI

# --- Настройка для Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Инициализируем клиент OpenAI с базовым URL Groq
client = None
if GROQ_API_KEY:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

# Используем быструю модель от Groq
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

app = Flask(__name__)

# --- Структура тем ---
THEME_SECTIONS = [
    {
        "title": "Базовые конструкции Python",
        "subtopics": [
            { "id": 'ввод и вывод данных, операции с числами и строками, форматирование', "name": 'Ввод и вывод данных. Операции с числами, строками. Форматирование' },
            { "id": 'условный оператор', "name": 'Условный оператор' },
            { "id": 'циклы', "name": 'Циклы' },
            { "id": 'вложенные циклы', "name": 'Вложенные циклы' },
        ]
    },
    {
        "title": "Коллекции и работа с памятью",
        "subtopics": [
            { "id": 'строки, кортежи, списки', "name": 'Строки, кортежи, списки' },
            { "id": 'множества, словари', "name": 'Множества, словари' },
            { "id": 'списочные выражения и модель памяти', "name": 'Списочные выражения. Модель памяти' },
            { "id": 'встроенные возможности коллекций', "name": 'Встроенные возможности по работе с коллекциями' },
            { "id": 'работа с файлами и json', "name": 'Потоковый ввод/вывод. Работа с файлами. JSON' },
        ]
    },
    {
        "title": "Функции и их особенности",
        "subtopics": [
            { "id": 'функции, области видимости, передача параметров', "name": 'Функции. Области видимости. Передача параметров' },
            { "id": 'позиционные и именованные аргументы, функции высших порядков, лямбда-функции', "name": 'Позиционные и именованные аргументы. Лямбда-функции' },
            { "id": 'рекурсия, декораторы, генераторы', "name": 'Рекурсия. Декораторы. Генераторы' },
        ]
    },
    {
        "title": "Объектно-ориентированное программирование",
        "subtopics": [
            { "id": 'классы, поля и методы', "name": 'Объектная модель Python. Классы, поля и методы' },
            { "id": 'волшебные методы и наследование', "name": 'Волшебные методы, переопределение методов. Наследование' },
            { "id": 'обработка исключений и модули', "name": 'Модель исключений Python. Try, except. Модули' },
        ]
    },
    {
        "title": "Библиотеки для обработки данных",
        "subtopics": [
            { "id": 'модули math и numpy', "name": 'Модули math и numpy' },
            { "id": 'модуль pandas', "name": 'Модуль pandas' },
            { "id": 'модуль requests', "name": 'Модуль requests' },
        ]
    }
]

# Запрещаем кэширование
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- ПРОМПТЫ ---
TASK_GENERATION_PROMPT_TEMPLATE = """
Выступайте в роли технического наставника по Python. Ваша задача — создать учебное задание.

Задание должно содержать:
1.  **title**: Название задания на русском языке.
2.  **task**: Четкое описание условия задачи на русском языке.
3.  **buggy_code**: Код функции на Python, который содержит **обязательную** и **неочевидную** логическую ошибку, связанную с темой "{theme}". Код не должен содержать синтаксических ошибок.

**ВАЖНОЕ ОГРАНИЧЕНИЕ:** В коде и условии задачи ЗАПРЕЩЕНО использовать следующие, более продвинутые концепции: {forbidden_themes}.

Ваш ответ должен быть представлен СТРОГО в формате JSON-строки (без дополнительных комментариев от себя, только валидный JSON).
"""

SOLUTION_VERIFICATION_PROMPT_TEMPLATE = """
Проанализируйте предоставленный код на соответствие техническому заданию. Пользователь уже выполнял проверку ранее, но продолжает работу/улучшение кода. Учти предыдущие попытки и оцени актуальный вариант.

**Техническое задание:** {task_description}
**Код для проверки:** ```python\n{user_code}\n```
**Ваша цель:** Определить, соответствует ли код заданию.
Ваш ответ должен быть представлен СТРОГО в формате JSON-строки со следующими ключами:
1.  **is_correct**: булево значение (true, если код полностью и корректно решает задачу; в противном случае — false).
2.  **explanation**: Суть ошибки или подтверждение корректности решения на русском языке.
"""

def query_groq_model(prompt):
    """Отправляет запрос к Groq API через OpenAI SDK и возвращает распарсенный JSON."""
    if not GROQ_API_KEY or not client:
        raise EnvironmentError("Не задан GROQ_API_KEY. Установите переменную окружения перед запуском.")

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        response_text = completion.choices[0].message.content
        if not response_text:
            raise Exception("Модель вернула пустой ответ.")
        
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        json_start = clean_text.find('{')
        json_end = clean_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            clean_json_str = clean_text[json_start:json_end]
        else:
            clean_json_str = clean_text

        return json.loads(clean_json_str.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Не удалось распарсить JSON из ответа модели: {e}\nОтвет: {response_text}")
    except Exception as e:
        raise Exception(f"Ошибка при работе с моделью через Groq API: {e}")

def get_forbidden_themes(current_theme_id):
    """Собирает список всех тем, которые идут после текущей."""
    forbidden = []
    found_current = False
    for section in THEME_SECTIONS:
        for subtopic in section['subtopics']:
            if found_current:
                forbidden.append(subtopic['name'])
            if subtopic['id'] == current_theme_id:
                found_current = True
    return ", ".join(forbidden) if forbidden else "нет"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-task')
def get_task():
    """Генерирует задачу по выбранной теме через Groq API."""
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY не задан. Установите переменную окружения перед запуском."}), 400

    selected_theme = request.args.get('theme', 'ввод и вывод данных, операции с числами и строками, форматирование')
    print(f"Запрошена тема: {selected_theme}")

    forbidden_themes_list = get_forbidden_themes(selected_theme)
    print(f"Запрещенные темы: {forbidden_themes_list}")

    try:
        print(f"Генерация задачи по теме '{selected_theme}' через Groq API...")
        generation_prompt = TASK_GENERATION_PROMPT_TEMPLATE.format(
            theme=selected_theme,
            forbidden_themes=forbidden_themes_list
        )
        task_data = query_groq_model(generation_prompt)

        if not all(k in task_data for k in ['task', 'buggy_code', 'title']):
            raise ValueError("Сгенерированные данные неполные (отсутствуют обязательные поля в JSON).")

        return jsonify(task_data)
    except Exception as e:
        print(f"Ошибка генерации Groq: {e}")
        return jsonify({"error": str(e)}), 500

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
        print("Запрос к Groq: проверка решения...")
        review_data = query_groq_model(verification_prompt)
        print("Результат проверки получен.")
        return jsonify(review_data)
    except Exception as e:
        print(f"Ошибка при проверке решения через Groq: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)