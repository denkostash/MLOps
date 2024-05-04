# Импортируем библиотеки
import os
import pandas as pd

# Создаем набор данных рост и вес человека
data1 = pd.DataFrame({
    "height": [170, 180, 190, 175, 185],
    "weight": [65, 70, 80, 72, 78]})

# Создаем набор данных площадь и цена дома
data2 = pd.DataFrame({
    "size": [1000, 1200, 1400, 1100, 1300],
    "price": [200000, 240000, 280000, 220000, 260000]})

# Создаем набор данных возраст и доход человека
data3 = pd.DataFrame({
    "age": [20, 25, 30, 23, 28],
    "money": [50000, 60000, 70000, 55000, 65000]})

# Создание папки data, если она не существует
os.makedirs('data', exist_ok=True)

# Сохраняем наборы данных в файл CSV
data1.to_csv("data/height_weight.csv", index=False)
data2.to_csv("data/size_price.csv", index=False)
data3.to_csv("data/age_money.csv", index=False)

