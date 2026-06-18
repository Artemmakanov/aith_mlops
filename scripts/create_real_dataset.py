import pandas as pd
from clearml import Dataset

# --- 1. Скачиваем реально существующий датасет ---
print("Скачиваем датасет Twitter Sentiment Analysis...")
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

# Читаем датасет (в оригинале он содержит ~32к строк)
df = pd.read_csv(url)

# Берем сбалансированную выборку 1000 строк (500 на класс, соотношение 1:1)
n_per_class = 500
small_df = pd.concat(
    [
        group.sample(n=n_per_class, random_state=42)
        for _, group in df.groupby('label')
    ]
).sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем локально в CSV
csv_filename = "real_sentiment_mini.csv"
small_df.to_csv(csv_filename, index=False)
print(f"Сохранен локальный файл {csv_filename} ({len(small_df)} строк).")

# --- 2. Создание ClearML Dataset через SDK ---
print("Создаем датасет в ClearML...")
dataset = Dataset.create(
    dataset_name='real_twitter_sentiment',
    dataset_project='students_project'
)

# --- 3. Загрузка файлов в Dataset ---
# Добавляем наш созданный csv
dataset.add_files(path=csv_filename)

# Загружаем файлы на файловый сервер ClearML
dataset.upload()

# --- 4. Фиксация версии ---
# ClearML автоматически присвоит версию (например, 1.0.0) и переведет датасет в статус "read-only".
dataset.finalize()

print("--------------------------------------------------")
print("Реальный датасет успешно загружен и зафиксирован в ClearML!")
print(f"Ваш dataset_id: {dataset.id}")
print("--------------------------------------------------")