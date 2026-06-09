from clearml import Task

# 1. Инициализируем задачу
# 'students_project' — название проекта, 'test_task' — имя эксперимента
task = Task.init(project_name='students_project', task_name='test_task_on_agent')

# 2. Указываем, что эту задачу нужно выполнить удаленно в очереди 'students'
task.execute_remotely(queue_name='students')

# --- Код, который будет выполняться на агенте ---
import time
print("Привет из контейнера! Задача успешно запущена агентом.")
time.sleep(10) # Имитация работы
print("Работа завершена.")