from clearml import Task, OutputModel, Model

# 1. Инициализируем задачу-регистратор (это отдельный служебный запуск)
task = Task.init(project_name='students_project', task_name='model_registry_publisher')

# 2. Укажите ID вашего ЛУЧШЕГО эксперимента из Этапа 2
BEST_TASK_ID = "6b35b136ff53459ebfc4136a0c55c82a"
best_task = Task.get_task(task_id=BEST_TASK_ID)

# Скачиваем файл модели (.pkl), который мы ранее сохранили как 'model_artifact'
print("Скачиваем артефакт лучшей модели...")
model_path = best_task.artifacts['model_artifact'].get_local_copy()

# 3. Вытаскиваем метрики из прошлого эксперимента
scalars = best_task.get_reported_scalars()
best_acc = scalars['Metrics']['Accuracy']['y'][-1]
best_f1 = scalars['Metrics']['F1 Score']['y'][-1]

print(f"Лучшие метрики -> Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")

# Логируем эти же метрики в текущую задачу, чтобы они "прилипли" к нашей новой модели
# и красиво отображались в таблице Registry
task.get_logger().report_scalar("Metrics", "Accuracy", value=best_acc, iteration=1)
task.get_logger().report_scalar("Metrics", "F1 Score", value=best_f1, iteration=1)

# 4. Создаем настоящую сущность Model в Registry
print("Регистрируем модель в Model Registry...")
output_model = OutputModel(
    task=task,
    name='sentiment_model_best',
    framework='scikit-learn'
)

# Передаем скачанный .pkl файл в Registry
output_model.update_weights(weights_filename=model_path)

# Добавляем теги и фиксируем версию 
output_model.tags = ['production', 'v1.0', 'sentiment-analysis']

# Запоминаем ID модели до того, как закроем задачу
model_id = output_model.id

# 5. Официальная публикация
# Сначала закрываем текущую таску, чтобы отвязать статус модели от статуса скрипта
task.close()

# Публикуем саму модель (фиксируем её, она становится Read-Only)
registered_model = Model(model_id=model_id)
registered_model.publish()

print("--------------------------------------------------")
print(f"✅ Модель успешно опубликована в Registry!")
print(f"✅ Model ID: {model_id}")
print("--------------------------------------------------")