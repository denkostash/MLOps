# MLOps. Практическое задание №2

Разработать собственный конвейер автоматизации для проекта машинного обучения.

## Структура файлов и каталогов

- data_creation.py - создает различные наборы данных, описывающие процесс изменения дневной температуры.
   - Функция для генерации данных
   - Генерация данных для обучения и тестирования
   - Сохранение данных в файлы

- model_preprocessing.py - выполняет предобработку данных с помощью sklearn.preprocessing.StandardScaler.
   - Загрузка данных
   - Предобработка данных
   - Сохранение предобработанных данных
     
- model_preparation.py - создает и обучает модель машинного обучения на построенных данных.
   - Загрузка предобработанных данных
   - Разделение данных на обучающую и валидационную выборки
   - Создание и обучение модели
   - Сохранение модели

- model_testing.py - проверяет модель машинного обучения на построенных данных.
   - Загрузка модели
   - Загрузка тестовых данных
   - Предсказание и оценка модели

- pipeline.sh - последовательно запускает все python-скрипты.
