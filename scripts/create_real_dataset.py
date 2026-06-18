import pandas as pd
from clearml import Dataset
from sklearn.model_selection import train_test_split

# --- 1. Скачиваем реально существующий датасет ---
print("Скачиваем датасет Twitter Sentiment Analysis...")
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

# Читаем датасет (в оригинале он содержит ~32к строк)
df = pd.read_csv(url)

# Берем стратифицированную выборку 500 строк, сохраняя пропорции классов
small_df, _ = train_test_split(
    df, train_size=1000, stratify=df['label'], random_state=42
)

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