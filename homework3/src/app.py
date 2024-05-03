import streamlit as st
import pandas as pd
import pickle
import json

# Фукнция для загрузки датасета
def load_model ():
    with open('/app/data/iris_mode.pkl', 'rb') as f:
        model = pickle.load(f)

model = load_model()

st.title ('Прогнозирование видов ириса')
st.write ('Пожалуйста, загрузите JSON-файл, содержащий признаки Iris для прогнозирования')

#Загрузка файлов пользователем
uploaded_file = st.file_uploader ('Выберите JSON-файл ', type = ['json'])
if uploaded_file is not None:

    data = json.load(uploaded_file)

    df = pd.DataFrame([data])

    try:
        #Предсказание
        prediction = model.predict(df['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        st.write ('Предполагаемые виды ирисов: ', prediction[0])
    except Exception as e:
        st.error (f"Ошибка в прогнозе: {e}")
    finally:
        uploaded_file.seek(0)