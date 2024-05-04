#Обучим модель линейной регрессии на датасете “рост_вес” и Создадим датасет с шумом в данных, добавив к целевой переменной случайный шум
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Загружаем датасет
data = pd.read_csv("data/height_weight.csv")

# Разделяем данные на признаки и целевую переменную
X = data.drop(columns=["weight"])
y = data["weight"]

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(X, y)

# Добавляем шум к целевой переменной
data["weight"] += np.random.normal(0, 10, len(data))

# Сохраняем датасет с шумом в файл CSV
data.to_csv("data/height_weight_noise.csv", index=False)
