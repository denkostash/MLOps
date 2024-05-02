# Импортируем библиотеки
import os
import numpy as np
import pandas as pd

# Создание наборов данных с различными характеристиками
# Данные по изменению дневной температуры
time = np.linspace(0, 1, 100)
data1 = pd.DataFrame({'time': time, 'temperature': 20 + 5 * np.sin(2 * np.pi * time)})
data2 = pd.DataFrame({'time': time, 'temperature': 20 + 5 * np.sin(2 * np.pi * time) + 2 * np.random.randn(100)})
data3 = pd.DataFrame({'time': time, 'temperature': 20 + 5 * np.sin(2 * np.pi * time) + 5 * np.random.randn(100)})
data4 = pd.DataFrame({'time': time, 'temperature': 20 + 5 * np.sin(2 * np.pi * time) + 10 * np.random.randn(100)})

# Разделение данных на train и test
train_data = [data1, data2]
test_data = [data3, data4]

# Создание папки train, если она не существует
os.makedirs('train', exist_ok=True)

# Создание папки test, если она не существует
os.makedirs('test', exist_ok=True)

# Сохранение данных train
for i, data in enumerate(train_data):
    data.to_csv(f'train/data{i}.csv', index=False)

# Сохранение данных test
for i, data in enumerate(test_data):
    data.to_csv(f'test/data{i}.csv', index=False)