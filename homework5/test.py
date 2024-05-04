import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загружаем 3 датасета с качественными данными и  1 шумовый датасет
data1 = pd.read_csv("data/height_weight.csv")
data2 = pd.read_csv("data/size_price.csv")
data3 = pd.read_csv("data/age_money.csv")
data4 = pd.read_csv("data/height_weight_noise.csv")

# Определяем тестовую функцию
def test_model_performance():
    # Создание модели линейной регрессии
    model = LinearRegression()

    # Обучаем модель на 1 датасете с качественными данными
    model.fit(data1["height"].values.reshape(-1, 1), data1["weight"])

    # Вычисление MSE на качественных данных
    mse_quality1 = np.mean((model.predict(data1["height"].values.reshape(-1, 1)) - data1["weight"]) ** 2)
    mse_quality2 = np.mean((model.predict(data2["size"].values.reshape(-1, 1)) - data2["price"]) ** 2)
    mse_quality3 = np.mean((model.predict(data3["age"].values.reshape(-1, 1)) - data3["money"]) ** 2)

    # Находим максимальный MSE среди качественных данных
    max_mse_quality = max(mse_quality1, mse_quality2, mse_quality3)

    # Вычисляем MSE на шумовом датасете
    mse_noise = np.mean((model.predict(data4["height"].values.reshape(-1, 1)) - data4["weight"]) ** 2)

    if mse_noise > max_mse_quality:
        print(f"Датасет 4 является шумовым, MSE: {mse_noise}")
    else:
        print(f"Датасет 4 не является шумовым, MSE: {mse_noise}")


    # Вывод MSE для каждого датасета
    print(f"MSE для датасета 1: {mse_quality1}")
    print(f"MSE для датасета 2: {mse_quality2}")
    print(f"MSE для датасета 3: {mse_quality3}")
    print(f"MSE для датасета 4: {mse_noise}")

test_model_performance()