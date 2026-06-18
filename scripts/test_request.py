import requests

# URL нашего эндпоинта (формируется из базового URL + /serve/ + имя_эндпоинта)
url = "http://localhost:8008/serve/sentiment"

# Подготовим примеры текстов (положительный и отрицательный классы)
# ClearML Serving для sklearn ожидает данные в ключе "x" в виде списка
payload = {
    "x": [
        "I absolutely love this product, it is amazing and wonderful!",
        "Terrible experience, very bad, worst purchase ever."
    ]
}

print(f"Отправляем POST запрос на {url} ...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("✅ Успешный ответ от сервера!")
    predictions = response.json()

    # Выводим результаты
    for text, label in zip(payload["x"], predictions):
        print(f"[{label}] <- '{text}'")
else:
    print(f"❌ Ошибка {response.status_code}: {response.text}")
