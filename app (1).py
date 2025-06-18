import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
    """
    Loads the pre-trained stacking classifier model and the RobustScaler.
    These files are expected to be in the same directory as the Streamlit app.
    """
    try:
        # Corrected file names to remove '(2)' based on common naming convention
        with open("stacking_classifier_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model atau file scaler tidak ditemukan. Pastikan 'stacking_classifier_model.pkl' dan 'scaler.pkl' berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# ===========================
# 2. Load Dataset
# ===========================
@st.cache_data
def load_data():
    """
    Loads the student lifestyle dataset, performs GPA categorization,
    and encodes stress levels and academic performance.
    """
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
        st.error("Error: File dataset 'student_lifestyle_dataset.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        st.stop()

data = load_data()

# Define the features used for the model
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
    X = data[features]
    y = data['Stress_Level_Encoded']

    # Check if scaler has been fitted before trying to access feature_names_in_
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    else:
        st.warning("Scaler belum di-fit. Melanjutkan dengan penskalaan langsung. Jika ini menyebabkan masalah, pastikan 'scaler.pkl' Anda di-fit dengan benar.")
        # If scaler wasn't fitted, X is already a DataFrame, no need to recreate
        pass 

    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
    except Exception as e:
        st.error(f"Error selama evaluasi model: {e}. Periksa apakah scaler dan model kompatibel dengan data.")
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
    report = classification_report(y_true_labels, y_pred_labels, target_names=['Low', 'Moderate', 'High'])
    st.text(report)

    st.subheader("📉 Kurva ROC dan AUC")
    n_classes = len(reverse_stress_mapping)
    y_binarize = label_binarize(y, classes=list(reverse_stress_mapping.keys()))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarize[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'orange', 'green']
    for i, color in zip(range(n_classes), colors):
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
    nama = st.text_input("Nama Anda:") # Moved from below for immediate input
    umur = st.number_input("Umur Anda:", min_value=5, max_value=100, value=20) # Moved from below

    study_hours = st.slider("Jam Belajar Per Hari", 0.0, 10.0, 5.0)
    sleep_hours = st.slider("Jam Tidur Per Hari", 0.0, 10.0, 7.0)
    physical_activity = st.slider("Jam Aktivitas Fisik Per Hari", 0.0, 13.0, 2.0)
    social_hours = st.slider("Jam Interaksi Sosial Per Hari", 0.0, 6.0, 2.0)
    extracurricular_input = st.selectbox("Ikut Ekstrakurikuler?", ["Ya", "Tidak"])
    extracurricular_hours = 1 if extracurricular_input == "Ya" else 0 # Corrected mapping
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

    # Create DataFrame for prediction
    # Ensure column order matches 'features' list used for training
    input_data = pd.DataFrame([[
        study_hours, sleep_hours, physical_activity,
        social_hours, extracurricular_hours, gpa, academic_performance_encoded
    ]], columns=features) # Use 'features' directly for columns

    if st.button("Prediksi Tingkat Stres"):
        if nama.strip() == "":
            st.error("❌ Mohon isi nama terlebih dahulu sebelum melakukan prediksi.")
        else:
            try:
                # Check if scaler has been fitted before trying to access feature_names_in_
                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                    # If scaler has feature_names_in_, ensure input_data columns match
                    # This line is redundant if input_data is already created with 'features' list as columns.
                    # It's safer to just transform if the column order is guaranteed to be correct from `features` list.
                    # input_data = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
                    pass
                else:
                    st.warning("Scaler belum di-fit. Melanjutkan dengan penskalaan langsung. Jika ini menyebabkan masalah, pastikan 'scaler.pkl' Anda di-fit dengan benar.")
                    pass

                # Scale the input data
                input_scaled = scaler.transform(input_data) # input_data already has correct features, no need to slice `[features]` again.
                
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
