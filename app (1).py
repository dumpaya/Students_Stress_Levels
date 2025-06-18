import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os # Import the os module for path handling

from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# ===========================
# 1. Load Model & Scaler
# ===========================
@st.cache_resource
def load_model_and_scaler():
    model_dir = "models"
    model_path = os.path.join(model_dir, "stacking_classifier_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            st.warning(f"Directory '{model_dir}' created. Please ensure your .pkl files are placed inside it.")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler files not found. Please ensure 'stacking_classifier_model.pkl' and 'scaler.pkl' are in the '{model_dir}' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

@st.cache_data

def load_data():
    try:
        df = pd.read_csv("student_lifestyle_dataset.csv")
        stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        performance_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

        df['Academic_Performance'] = df['GPA'].apply(
            lambda x: 'Excellent' if x >= 3.5 else 'Good' if x >= 3.0 else 'Fair' if x >= 2.0 else 'Poor'
        )
        df['Academic_Performance_Encoded'] = df['Academic_Performance'].map(performance_mapping)
        df['Stress_Level_Encoded'] = df['Stress_Level'].map(stress_mapping)

        return df
    except FileNotFoundError:
        st.error("Error: Dataset file 'student_lifestyle_dataset.csv' not found. Please ensure it's in the same directory as the Streamlit app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

data = load_data()

features = [
    "Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "Physical_Activity_Hours_Per_Day",
    "Social_Hours_Per_Day", "Extracurricular_Hours_Per_Day", "GPA", "Academic_Performance_Encoded"
]

# ===========================
# 3. Sidebar Navigation
# ===========================
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih halaman", ["Deskripsi Data", "Evaluasi Model", "Prediksi", "Anggota Kelompok"])
st.sidebar.write(f"Halaman terpilih: `{page}`")

# ===========================
# 4. Deskripsi Data
# ===========================
if page == "Deskripsi Data":
    st.title("📊 Deskripsi Dataset")
    st.markdown("""
    ## 🗂️ Dataset Overview: Student Lifestyle and Stress
    
    Dataset ini berisi informasi gaya hidup mahasiswa dan hubungannya dengan tingkat stres serta performa akademik.
    
    ### 🎯 Tujuan
    Memprediksi **Tingkat Stres** berdasarkan atribut gaya hidup dan akademik.
    
    ### 🔑 Fitur Utama
    - **Jumlah data:** 2.000 mahasiswa
    - **Kolom:** 8 fitur + target
    - **Fitur gaya hidup:**
      - 🕒 `Study_Hours_Per_Day`: Jam belajar per hari
      - 😴 `Sleep_Hours_Per_Day`: Jam tidur per hari
      - 🏃‍♂️ `Physical_Activity_Hours_Per_Day`: Aktivitas fisik harian
      - 🗣️ `Social_Hours_Per_Day`: Interaksi sosial
      - 🎭 `Extracurricular_Hours_Per_Day`: Kegiatan ekstrakurikuler
    - **Akademik & Target:**
      - 🎓 `GPA`: Nilai rata-rata akademik
      - ⚡ `Stress_Level`: Target prediksi — Rendah, Sedang, Tinggi
    
    ### 📌 Insight Data
    - Stres tinggi → Jam belajar tinggi & tidur rendah
    - Stres rendah → Aktivitas fisik & sosial seimbang
    - Fitur paling berpengaruh: **Jam Belajar** & **Jam Tidur**
    """)

    st.dataframe(data.head())

    st.subheader("Distribusi Kelas Tingkat Stres")
    st.markdown("""
    ### 📊 Penjelasan Distribusi Kelas Tingkat Stres
    
    Distribusi ini menunjukkan **jumlah mahasiswa** dalam setiap kategori stres:
    
    - 🟢 **Rendah**: Mahasiswa yang memiliki gaya hidup seimbang—cukup tidur, waktu belajar moderat, dan aktif secara fisik & sosial.
    - 🟡 **Sedang**: Umumnya memiliki tekanan akademik atau waktu belajar tinggi, namun masih menjaga keseimbangan aktivitas lainnya.
    - 🔴 **Tinggi**: Cenderung disebabkan oleh jam belajar berlebihan, kurang tidur, dan minim aktivitas sosial atau fisik.
    
    Distribusi kelas ini penting karena:
    - Memberi gambaran apakah data seimbang atau tidak.
    - Mempengaruhi performa model prediksi (model bisa bias jika mayoritas data berasal dari satu kelas).
    
    💡 Jika proporsi kelas tidak seimbang (misalnya sebagian besar Sedang), teknik seperti **SMOTE** digunakan saat pelatihan untuk menyeimbangkan data.
    """)
    st.bar_chart(data["Stress_Level"].value_counts())
    st.markdown("""
    Distribusi ini menunjukkan jumlah mahasiswa yang tergolong dalam tiga tingkat stres:
    - **Tinggi**: 1029 mahasiswa (**51.5%**)
    - **Sedang**: 674 mahasiswa (**33.7%**)
    - **Rendah**: 297 mahasiswa (**14.9%**)
    
    Distribusi ini menunjukkan bahwa **lebih dari setengah mahasiswa mengalami stres tinggi**, yang mungkin berkaitan dengan tekanan akademik, kurang tidur, atau kebiasaan gaya hidup yang tidak seimbang.
    """)

    st.subheader("Diagram Lingkaran Tingkat Stres")
    fig, ax = plt.subplots()
    data["Stress_Level"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax, shadow=True, explode=[0.05]*3
    )
    ax.set_ylabel("")
    st.pyplot(fig)


    st.subheader("📈 Korelasi Antar Fitur")
    st.markdown("""
    Korelasi digunakan untuk mengetahui hubungan antara fitur gaya hidup dengan **IPK** atau **Tingkat Stres**:
    
    - Nilai **positif** → Hubungan searah (jika fitur naik, target naik).
    - Nilai **negatif** → Hubungan berlawanan (jika fitur naik, target turun).
    
    Nilai korelasi mendekati 1 atau -1 menunjukkan hubungan yang kuat.
    """)
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[features + ["Stress_Level_Encoded"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
    
    st.subheader("📊 Distribusi IPK Mahasiswa")
    fig_gpa, ax = plt.subplots()
    sns.histplot(data["GPA"], bins=20, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("IPK")
    ax.set_ylabel("Jumlah Mahasiswa")
    st.pyplot(fig_gpa)

    st.subheader("🛌 Jam Tidur vs Tingkat Stres")
    st.markdown("Boxplot ini menunjukkan hubungan antara **lama tidur** dan **tingkat stres mahasiswa**.")
    fig_sleep, ax = plt.subplots()
    sns.boxplot(x="Stress_Level", y="Sleep_Hours_Per_Day", data=data, palette="Set2", ax=ax)
    st.pyplot(fig_sleep)

    st.subheader("📚 Jam Belajar vs IPK")
    st.markdown("Scatterplot untuk melihat apakah semakin banyak jam belajar selalu meningkatkan IPK.")
    fig_study_gpa, ax = plt.subplots()
    sns.scatterplot(x="Study_Hours_Per_Day", y="GPA", hue="Stress_Level", data=data, palette="Set1", ax=ax)
    st.pyplot(fig_study_gpa)

# ===========================
# 5. Evaluasi Model
# ===========================
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model")
    st.markdown("""
    ### 🧪 Evaluasi Model yang Digunakan
    
    Model yang digunakan adalah **Stacking Classifier**, yaitu gabungan dari beberapa model dasar (XGBoost, Logistic Regression, Decision Tree, Random Forest, dan SVM) yang dipadukan menggunakan meta-model Random Forest.
    Untuk menilai performa model, berikut metrik evaluasi yang digunakan:
    
    - **🎯 Akurasi**: Proporsi data uji yang berhasil diprediksi dengan benar. Metrik ini memberikan gambaran umum seberapa sering model membuat prediksi yang benar.
    
    - **📊 Confusion Matrix**: Menunjukkan perbandingan antara label sebenarnya dan hasil prediksi. Memudahkan untuk melihat kesalahan spesifik antar kelas stres (Rendah, Sedang, Tinggi).
    
    - **📉 Kurva ROC dan AUC (Area Under Curve)**:
      - ROC (Receiver Operating Characteristic) menunjukkan trade-off antara Tingkat True Positive dan Tingkat False Positive.
      - AUC mengukur kemampuan model membedakan antara kelas: semakin tinggi (mendekati 1), semakin baik performa model.
    
    - **🧾 Laporan Klasifikasi**:
      - **Presisi**: Seberapa akurat model saat memprediksi suatu kelas.
      - **Recall**: Seberapa baik model mendeteksi semua instance dari suatu kelas.
      - **F1-Score**: Harmonic mean dari presisi dan recall, menggambarkan keseimbangan antara keduanya.
    
    Evaluasi dilakukan menggunakan **data asli** yang telah diseimbangkan menggunakan **SMOTE** dan dinormalisasi dengan **RobustScaler**, agar hasil prediksi lebih adil dan tidak bias terhadap kelas dominan.
    """)

    # Prepare data for evaluation
    X = data[features].copy() # Use .copy() to avoid SettingWithCopyWarning
    y = data['Stress_Level_Encoded']

    # Ensure X has the same column names and order as expected by the scaler
    # The scaler.feature_names_in_ attribute stores the feature names seen during fit.
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        # Reorder columns of X to match scaler's expected order
        try:
            X = X[scaler.feature_names_in_]
        except KeyError as e:
            st.error(f"Error: Column mismatch between data and scaler's fitted features. Missing column: {e}. Please ensure your dataset contains all features used during model training and scaling.")
            st.stop()
    else:
        st.warning("Scaler does not have 'feature_names_in_'. Proceeding with direct scaling. Ensure column order of input data matches training data.")
        # If the scaler doesn't have feature_names_in_, it means it was loaded
        # from a .pkl file that was not properly fitted to retain this attribute.
        # In this case, we trust that 'features' list correctly represents the order.
        pass # X is already a DataFrame with 'features' columns in the correct order

    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
    except Exception as e:
        st.error(f"Error during model evaluation: {e}. Please check if the scaler and model are compatible with the data.")
        st.stop()
    
    # Mapping for display
    reverse_stress_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
    y_true_labels = y.map(reverse_stress_mapping)
    y_pred_labels = pd.Series(y_pred).map(reverse_stress_mapping)

    st.subheader("🎯 Akurasi Model")
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Akurasi: {accuracy:.4f}")

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['Low', 'Moderate', 'High'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Moderate', 'High'])
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    plt.title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("🧾 Laporan Klasifikasi")
    report = classification_report(y, y_pred, target_names=['Low', 'Moderate', 'High'], output_dict=True)
    st.dataframe(pd.DataFrame(report).T)


    st.subheader("📉 Kurva ROC dan AUC")
    n_classes = len(reverse_stress_mapping)
    y_binarize = label_binarize(y, classes=list(reverse_stress_mapping.keys()))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Ensure y_pred_proba has the correct number of columns (n_classes)
    if y_pred_proba.shape[1] != n_classes:
        st.warning("y_pred_proba has an unexpected number of columns for ROC curve plotting. Skipping ROC plot.")
    else:
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        colors = ['blue', 'orange', 'green']
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_binarize[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Kelas {reverse_stress_mapping[i]} (AUC = {roc_auc[i]:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('Tingkat False Positive')
        ax_roc.set_ylabel('Tingkat True Positive')
        ax_roc.set_title('Kurva Receiver Operating Characteristic (ROC)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.subheader("🧠 Ringkasan Hasil Pelatihan Model")
    st.markdown("""
    Model dilatih menggunakan pendekatan **Stacking Classifier**, yaitu menggabungkan beberapa algoritma dasar yang kuat dengan meta-learner.  
    Berikut adalah detail proses pelatihan model:
    
    ---
    
    #### 🔢 Dataset:
    - Jumlah data: **2.000 mahasiswa**
    - Jumlah fitur: **8 kolom** (gabungan numerik dan kategorik)
    - Fitur numerik termasuk: **Study, Sleep, GPA, dsb**
    - Target: `Stress_Level` (Low, Moderate, High)
    
    ---
    
    #### 📊 Praproses:
    - Konversi label target menjadi numerik:  
      `Low = 0`, `Moderate = 1`, `High = 2`
    - Normalisasi fitur numerik dengan **RobustScaler**
    - Penyeimbangan kelas target menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** > Teknik ini efektif mengatasi dominasi label tertentu (misal `High`) agar model tidak bias.
    
    ---
    
    #### ⚙️ Arsitektur Model:
    - **Base Learners**:
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - XGBoost Classifier
    - **Meta-Learner**:
        - Random Forest Classifier  
          > Digunakan untuk menggabungkan hasil prediksi dari base learners.
    
    ---
    
    #### 📈 Evaluasi Training:
    - **Akurasi Pelatihan**: 100%
    - **Confusion Matrix**: Semua prediksi benar
    - **ROC AUC Score**: 1.00 untuk semua kelas
    - **Classification Report**:
        - Precision: 1.00
        - Recall: 1.00
        - F1-score: 1.00
    
    > 💡 Performa sempurna di data pelatihan menunjukkan bahwa model **fit sangat baik**, namun perlu diuji lebih lanjut menggunakan data uji atau validasi silang untuk mengecek kemungkinan **overfitting**.
    """)


# ===========================
# 6. Prediction
# ===========================
elif page == "Prediksi":
    st.title("🔮 Prediksi Tingkat Stres Mahasiswa")
    st.markdown("""
    ### Masukkan data gaya hidup dan akademik mahasiswa untuk memprediksi tingkat stres.
    """)

    # Input fields for user
    st.subheader("Masukkan Atribut Gaya Hidup dan Akademik:")
    nama = st.text_input("Nama Anda:")
    umur = st.number_input("Umur Anda:", min_value=5, max_value=100, value=20)

    study_hours = st.slider("Jam Belajar Per Hari", 0.0, 10.0, 5.0)
    sleep_hours = st.slider("Jam Tidur Per Hari", 0.0, 10.0, 7.0)
    physical_activity = st.slider("Jam Aktivitas Fisik Per Hari", 0.0, 13.0, 2.0)
    social_hours = st.slider("Jam Interaksi Sosial Per Hari", 0.0, 6.0, 2.0)
    extracurricular_input = st.selectbox("Ikut Ekstrakurikuler?", ["Ya", "Tidak"])
    extracurricular_hours = 1 if extracurricular_input == "Ya" else 0
    gpa = st.slider("IPK", 0.0, 4.0, 3.0)

    # Encode Academic_Performance based on GPA for prediction
    def encode_academic_performance(gpa_val):
        if gpa_val >= 3.5:
            return 3  # Excellent
        elif gpa_val >= 3.0:
            return 2  # Good
        elif gpa_val >= 2.0:
            return 1  # Fair
        else:
            return 0  # Poor
    
    academic_performance_encoded = encode_academic_performance(gpa)

    # Create DataFrame for prediction, ensuring column order matches 'features'
    input_data = pd.DataFrame([[
        study_hours, sleep_hours, physical_activity,
        social_hours, extracurricular_hours, gpa, academic_performance_encoded
    ]], columns=features) # Use 'features' directly as columns

    if st.button("Prediksi Tingkat Stres"):
        if nama.strip() == "":
            st.error("❌ Mohon isi nama terlebih dahulu sebelum melakukan prediksi.")
        else:
            try:
                # Ensure input_data columns match scaler's expected order
                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                    # Reorder input_data columns to match scaler's expected order
                    # This step is crucial if the scaler.pkl was fitted on a DataFrame
                    # and retains feature_names_in_
                    input_data_ordered = input_data[scaler.feature_names_in_]
                else:
                    # If scaler doesn't have feature_names_in_, trust the 'features' list order
                    input_data_ordered = input_data.copy()
                    st.warning("Scaler does not have 'feature_names_in_'. Assuming the input feature order is correct.")
                
                # Scale the input data
                input_scaled = scaler.transform(input_data_ordered)
                
                # Make prediction
                prediction_encoded = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]

                # Decode prediction
                reverse_stress_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
                predicted_stress_level = reverse_stress_mapping[prediction_encoded]

                st.subheader("Hasil Prediksi:")
                st.write(f"Hai {nama} (umur {umur}), tingkat stresmu diprediksi: **{predicted_stress_level}**")
                
                st.subheader("Probabilitas Prediksi per Kelas:")
                proba_df = pd.DataFrame({
                    'Tingkat Stres': ['Low', 'Moderate', 'High'],
                    'Probabilitas': prediction_proba
                })
                st.bar_chart(proba_df.set_index('Tingkat Stres'))

                if predicted_stress_level == 'High':
                    st.warning("Tingkat stres yang diprediksi adalah TINGGI. Pertimbangkan untuk menyesuaikan gaya hidup.")
                elif predicted_stress_level == 'Moderate':
                    st.info("Tingkat stres yang diprediksi adalah SEDANG. Perhatikan keseimbangan gaya hidup Anda.")
                else:
                    st.success("Tingkat stres yang diprediksi adalah RENDAH. Pertahankan gaya hidup seimbang Anda!")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# ===========================
# 7. Anggota Kelompok
# ===========================
elif page == "Anggota Kelompok":
    st.title("👥 Kelompok 4")
    st.markdown("""
    ## Anggota Kelompok:

    1. 👩‍🎓 **Hanny Wahyu Khairuni (2304030050)**  
    2. 👩‍🎓 **Alya Siti Fathimah (2304030058)**  
    3. 👨‍🎓 **Alfian Noor Khoeruddin (2304030070)**  
    4. 👩‍🎓 **Arini Salmah (2304030080)**
    """)
