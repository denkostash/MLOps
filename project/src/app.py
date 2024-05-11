import streamlit as st
import pandas as pd
import pickle
import json
from palmerpenguins import load_penguins

# Фукнция для загрузки датасета
def load_model ():
    with open('data/penguins.pkl', 'rb') as f:
        model = pickle.load(f)

model = load_model()

st.title ('Прогнозирование вида пингвина')
st.write ('Пожалуйста, загрузите JSON-файл, содержащий признаки пингвинов для прогнозирования')

#Загрузка файлов пользователем
uploaded_file = st.file_uploader ('Выберите JSON-файл ', type = ['json'])
if uploaded_file is not None:

    data = json.load(uploaded_file)

    df = pd.DataFrame([data])

    try:
        #Предсказание
        prediction = model.predict(df['1']) # ???????????????????????????????
        st.write ('Предполагаемый вид пингвина: ', prediction[0])
    except Exception as e:
        st.error (f"Ошибка в прогнозе: {e}")
    finally:
        uploaded_file.seek(0)