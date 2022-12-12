import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
from pickle import dump
import joblib
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score


st.title("Project UAS DATA MINING")
st.write("Oleh : Sulimah | 200411100054")
# with st.sidebar:
selected = option_menu(
  menu_title  = None,
  options     = ["Import Data","Preprocessing","Modeling","Implementation"],
  icons       = ["data","process","model","implemen"],
  orientation = "horizontal",
)

df_train = pd.read_csv("heart.csv")
y = df_train['output']


# View Data
if(selected == "Import Data"):
  st.write("# Deskripsi")
  st.write("Data yang digunakan adalah data breast cancer :")
  st.write("https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")
  
  st.markdown("---")
  st.write("# Import Data")
  st.dataframe(df_train)


# Preprocessing
elif(selected == 'Preprocessing'):
  st.write("# Preprocessing")
  st.write("### Normalisasi")
  st.write('Melakukan Normalisasi pada semua fitur dan mengambil fitur yang memiliki tipe data numerik')
  st.dataframe(df_train)

  st.markdown("---")
  st.write("### Data yang telah dinormalisasi")
  st.write('Fitur numerikal sudah dinormalisasi')
  scaler = MinMaxScaler()
  df_train_pre = scaler.fit_transform(df_train.drop(columns=['output']))
  st.dataframe(df_train_pre)

  # Save Scaled
  joblib.dump(df_train_pre, 'model/df_train_pre.sav')
  joblib.dump(scaler,'model/df_scaled.sav')


# Modeling
elif(selected == 'Modeling'):
  st.write("# Modeling")
  # knn, nb, dtc = st.tabs(['K-NN', 'Naive-Bayes', 'Decission Tree'])
  st.write("Sistem ini menggunakan 3 modeling yaitu KNN, Naive-Bayes, dan Decission Tree")
  knn_cekbox = st.checkbox("KNN")
  bayes_gaussian_cekbox = st.checkbox("Naive-Bayes Gaussian")
  decission3_cekbox = st.checkbox("Decission Tree")

  #===================== Cek Box ====================
  if knn_cekbox:
    st.write("##### KNN")
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    scores = {}
    for i in range(1, 20+1):
      KN = KNeighborsClassifier(n_neighbors = i)
      KN.fit(x_train, y_train)
      y_pred = KN.predict(x_test)
      scores[i] = accuracy_score(y_test, y_pred)
        
    best_k = max(scores, key=scores.get)
    st.warning("Dengan menggunakan metode KNN didapatkan akurasi sebesar:")
    st.warning(f"Akurasi = {max(scores.values())* 100}%")
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)

    # Save Model
    joblib.dump(knn, 'model/knn_model.sav') # Menyimpan Model ke dalam folder model
    # st.write(df_train_pre)

    # st.warning("Dengan menggunakan metode KNN didapatkan akurasi sebesar:")
    # st.warning(f"Akurasi  =  {knn_accuracy}%")
    st.markdown("---")

  if bayes_gaussian_cekbox:
    st.write("##### Naive Bayes Gausssian")
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    # Save Model
    joblib.dump(nb, 'model/nb_model.sav') # Menyimpan Model ke dalam folder model
    
    y_pred = nb.predict(x_test)
    akurasi = accuracy_score(y_test,y_pred)
    
    st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
    st.info(f'Akurasi = {akurasi*100}%')
    # st.write(df_train_pre)

    # st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
    # st.info(f"Akurasi = {gauss_accuracy}%")
    st.markdown("---")

  if decission3_cekbox:
    st.write("##### Decission Tree")
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    # Save Model
    joblib.dump(dtc, 'model/dtc_model.sav') # Menyimpan Model ke dalam folder model
    
    y_pred = dtc.predict(x_test)
    akurasi = accuracy_score(y_test,y_pred)
    
    st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
    st.success(f'Akurasi = {akurasi*100}%')
    # st.write(df_train_pre)

    # st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
    # st.success(f"Akurasi = {decission3_accuracy}%")


# Implementasi
elif(selected == 'Implementation'):
  st.write("# Implementation")
  st.write("##### Input fitur")

  age = st.number_input("Masukkan Umur", min_value=29, max_value=77)
  gender = st.number_input("Masukkan Jenis Kelamin", min_value=0, max_value=1)
  chest_pain = st.number_input("Masukkan Type Nyeri Dada", min_value=0, max_value=3)
  blood_pressure = st.number_input("Masukkan Tekanan Darah (mm/Hg)", min_value=94, max_value=200)
  cholestoral = st.number_input("Masukkan Kadar Kolesterol (mm/dl)", min_value=126, max_value=564)
  heart_rate = st.number_input("Masukkan Detak Jantung Maximal", min_value=71, max_value=202)
  oldpeak = st.number_input("Masukkan Oldpeak", min_value=0.0, max_value=6.2)

  cek_knn = st.button('Cek Pasien')
  inputan = [[age, gender, chest_pain, blood_pressure, cholestoral, heart_rate, oldpeak]]
  
  scaler = joblib.load('model/df_scaled.sav')
  data_scaler = scaler.transform(inputan)

  FIRST_IDX = 0
  k_nn = joblib.load("model/knn_model.sav")
  if cek_knn:
    hasil_test = k_nn.predict(data_scaler)[FIRST_IDX]
    if hasil_test == 0:
      st.success(f'Pasien Tidak Mengidap Serangan Jantung Berdasarkan Metode K-NN')
    else:
      st.error(f'Pasien Mengidap Serangan Jantung Berdasarkan Metode K-NN')
