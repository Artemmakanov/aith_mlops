import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from clearml import Task, Dataset

# --- 1. Создание ClearML Task ---
task = Task.init(project_name='students_project', task_name='sentiment_training')

# --- 2. Логирование гиперпараметров ---
# Словарь параметров автоматически появится в Web UI, и мы сможем менять его при перезапусках
parameters = {
    'max_features': 1000,
    'C': 1.0,
    'test_size': 0.2,
    'dataset_id': 'c2ab0b80726043049b68f80ac79c2086' 
}
task.connect(parameters)

# --- 3. Отправка в очередь ---
# Код ниже этой строчки локально не выполнится. Задача уйдет в очередь, и её заберет Агент.
task.execute_remotely(queue_name='students')

# --- 4. Использование Dataset из ClearML ---
print("Скачиваем датасет через агента...")
dataset_path = Dataset.get(dataset_id=parameters['dataset_id']).get_local_copy()

# В датасете Twitter Sentiment Analysis тексты лежат в 'tweet', а таргет в 'label'
df = pd.read_csv(f"{dataset_path}/real_sentiment_mini.csv")
X = df['tweet'].fillna('')
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], random_state=42)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=parameters['max_features'])
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
print("Обучаем модель...")
model = LogisticRegression(C=parameters['C'], random_state=42)
model.fit(X_train_vec, y_train)
preds = model.predict(X_test_vec)

# --- 5. Логирование метрик (Accuracy, F1) ---
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

task.get_logger().report_scalar(title='Metrics', series='Accuracy', value=acc, iteration=1)
task.get_logger().report_scalar(title='Metrics', series='F1 Score', value=f1, iteration=1)
print(f"Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

# --- 6. Логирование Confusion Matrix как изображения ---
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")

# Явно отправляем график в ClearML
task.get_logger().report_matplotlib_figure(
    title="Confusion Matrix",
    series="Test Data",
    iteration=1,
    figure=fig
)

# --- 7. Сохранение модели как Artifact ---
joblib.dump(model, 'sentiment_model.pkl')
task.upload_artifact(name='model_artifact', artifact_object='sentiment_model.pkl')

print("Обучение успешно завершено. Артефакты отправлены на сервер.")