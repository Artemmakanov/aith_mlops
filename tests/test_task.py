from clearml import Task

# 1. Инициализируем задачу
task = Task.init(project_name='students_project', task_name='test_task_on_agent')

# 2. Указываем конкретную ветку
# ClearML автоматически найдет репозиторий, если скрипт в нем лежит
task.set_repo(
    repo='https://github.com/Artemmakanov/aith_mlops.git',
    branch='step_0',
)

# 3. Отправляем в очередь
task.execute_remotely(queue_name='students')

# --- Код, который будет выполняться на агенте ---
import time
print("Привет из контейнера! Задача успешно запущена агентом.")
time.sleep(10) # Имитация работы
print("Работа завершена.")