# Импортируем библиотеки
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Предобработка данных
def preprocess_data(data):
    scaler = StandardScaler()
    data['temperature'] = scaler.fit_transform(data['temperature'].values.reshape(-1, 1))
    return data

# Предобработка данных train
train_data = []
for i in range(len(os.listdir('train'))):
    data = pd.read_csv(f'train/data{i}.csv')
    train_data.append(preprocess_data(data))

# Предобработка данных test
test_data = []
for i in range(len(os.listdir('test'))):
    data = pd.read_csv(f'test/data{i}.csv')
    test_data.append(preprocess_data(data))

# Создание папки train, если она не существует
os.makedirs('train/train_preprocessed', exist_ok=True)

# Создание папки test, если она не существует
os.makedirs('test/test_preprocessed', exist_ok=True)

# Сохранение предобработанных данных train
for i, data in enumerate(train_data):
    data.to_csv(f'train/train_preprocessed/preprocessed_data{i}.csv', index=False)

# Сохранение предобработанных данных test
for i, data in enumerate(test_data):
    data.to_csv(f'test/test_preprocessed/preprocessed_data{i}.csv', index=False)