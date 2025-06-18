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
    with open("stacking_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ===========================
# 2. Load Dataset
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("student_lifestyle_dataset.csv")
    stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    performance_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

    df['Academic_Performance'] = df['GPA'].apply(
        lambda x: 'Excellent' if x >= 3.5 else 'Good' if x >= 3.0 else 'Fair' if x >= 2.0 else 'Poor'
    )
    df['Academic_Performance_Encoded'] = df['Academic_Performance'].map(performance_mapping)
    df['Stress_Level_Encoded'] = df['Stress_Level'].map(stress_mapping)
    
    return df

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
st.sidebar.write(f"Page terpilih: `{page}`")

# ===========================
# 4. Deskripsi Data
# ===========================
if page == "Deskripsi Data":
    st.title("📊 Deskripsi Dataset")
    st.markdown("""
    ## 🗂️ Dataset Overview: Student Lifestyle and Stress
    
    Dataset ini berisi informasi gaya hidup mahasiswa dan hubungannya dengan tingkat stres serta performa akademik.
    
    ### 🎯 Tujuan
    Memprediksi **Stress Level** berdasarkan atribut gaya hidup dan akademik.
    
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
      - ⚡ `Stress_Level`: Target prediksi — Low, Moderate, High
    
    ### 📌 Insight Data
    - Stres tinggi → Jam belajar tinggi & tidur rendah
    - Stres rendah → Aktivitas fisik & sosial seimbang
    - Fitur paling berpengaruh: **Study Hours** & **Sleep Hours**
    """)

    st.dataframe(data.head())

    st.subheader("Distribusi Kelas Stress Level")
    st.markdown("""
    ### 📊 Penjelasan Distribusi Kelas Stress Level
    
    Distribusi ini menunjukkan **jumlah mahasiswa** dalam setiap kategori stres:
    
    - 🟢 **Low**: Mahasiswa yang memiliki gaya hidup seimbang—cukup tidur, waktu belajar moderat, dan aktif secara fisik & sosial.
    - 🟡 **Moderate**: Umumnya memiliki tekanan akademik atau waktu belajar tinggi, namun masih menjaga keseimbangan aktivitas lainnya.
    - 🔴 **High**: Cenderung disebabkan oleh jam belajar berlebihan, kurang tidur, dan minim aktivitas sosial atau fisik.
    
    Distribusi kelas ini penting karena:
    - Memberi gambaran apakah data seimbang atau tidak.
    - Mempengaruhi performa model prediksi (model bisa bias jika mayoritas data berasal dari satu kelas).
    
    💡 Jika proporsi kelas tidak seimbang (misalnya sebagian besar Moderate), teknik seperti **SMOTE** digunakan saat training untuk menyeimbangkan data.
    """)
    st.bar_chart(data["Stress_Level"].value_counts())
    st.markdown("""
    Distribusi ini menunjukkan jumlah mahasiswa yang tergolong dalam tiga tingkat stress:
    - **High** (Tinggi): 1029 mahasiswa (**51.5%**)
    - **Moderate** (Sedang): 674 mahasiswa (**33.7%**)
    - **Low** (Rendah): 297 mahasiswa (**14.9%**)
    
    Distribusi ini menunjukkan bahwa **lebih dari setengah mahasiswa mengalami stres tinggi**, yang mungkin berkaitan dengan tekanan akademik, kurang tidur, atau kebiasaan gaya hidup yang tidak seimbang.
    """)

    st.subheader("Diagram Pie Stress Level")
    fig, ax = plt.subplots()
    data["Stress_Level"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax, shadow=True, explode=[0.05]*3
    )
    ax.set_ylabel("")
    st.pyplot(fig)


    st.subheader("📈 Korelasi Antar Fitur")
    st.markdown("""
    Korelasi digunakan untuk mengetahui hubungan antara fitur gaya hidup dengan **GPA** atau **Stress Level**:
    
    - Nilai **positif** → Hubungan searah (jika fitur naik, target naik).
    - Nilai **negatif** → Hubungan berlawanan (jika fitur naik, target turun).
    
    Nilai korelasi mendekati 1 atau -1 menunjukkan hubungan yang kuat.
    """)
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[features + ["Stress_Level_Encoded"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
    
    st.subheader("📊 Distribusi GPA Mahasiswa")
    fig_gpa, ax = plt.subplots()
    sns.histplot(data["GPA"], bins=20, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("GPA")
    ax.set_ylabel("Jumlah Mahasiswa")
    st.pyplot(fig_gpa)

    st.subheader("🛌 Jam Tidur vs Tingkat Stres")
    st.markdown("Boxplot ini menunjukkan hubungan antara **lama tidur** dan **tingkat stres mahasiswa**.")
    fig_sleep, ax = plt.subplots()
    sns.boxplot(x="Stress_Level", y="Sleep_Hours_Per_Day", data=data, palette="Set2", ax=ax)
    st.pyplot(fig_sleep)

    st.subheader("📚 Jam Belajar vs GPA")
    st.markdown("Scatterplot untuk melihat apakah semakin banyak jam belajar selalu meningkatkan GPA.")
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
    
    - **📊 Confusion Matrix**: Menunjukkan perbandingan antara label sebenarnya dan hasil prediksi. Memudahkan untuk melihat kesalahan spesifik antar kelas stres (Low, Moderate, High).
    
    - **📉 ROC Curve dan AUC (Area Under Curve)**:
      - ROC (Receiver Operating Characteristic) menunjukkan trade-off antara True Positive Rate dan False Positive Rate.
      - AUC mengukur kemampuan model membedakan antara kelas: semakin tinggi (mendekati 1), semakin baik performa model.
    
    - **🧾 Classification Report**:
      - **Precision**: Seberapa akurat model saat memprediksi suatu kelas.
      - **Recall**: Seberapa baik model mendeteksi semua instance dari suatu kelas.
      - **F1-Score**: Harmonic mean dari precision dan recall, menggambarkan keseimbangan antara keduanya.
    
    Evaluasi dilakukan menggunakan **data asli** yang telah diseimbangkan menggunakan **SMOTE** dan dinormalisasi dengan **RobustScaler**, agar hasil prediksi lebih adil dan tidak bias terhadap kelas dominan.
    """)


    X = data[features]
    X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    X_scaled = scaler.transform(X)

    y = data["Stress_Level_Encoded"]
    class_labels = ["Low", "Moderate", "High"]
    
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    acc = accuracy_score(y, y_pred)

    st.subheader("🎯 Akurasi")
    st.success(f"Akurasi: {acc * 100:.2f}%")
    st.markdown("""
    **Akurasi** adalah persentase prediksi yang benar dari keseluruhan data.
    
    > **Rumus**: (Jumlah Prediksi Benar) / (Total Data)
    
    - Jika akurasi = 100%, artinya semua prediksi tepat.
    - ⚠️ Akurasi tinggi **tidak selalu berarti model bagus**, apalagi jika distribusi kelas tidak seimbang (misalnya mayoritas data ada di kelas "High").
    """)

    st.subheader("📊 Confusion Matrix")
    fig_cm, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=class_labels, ax=ax)
    st.pyplot(fig_cm)
    st.markdown("""
    **Confusion Matrix** adalah tabel yang membandingkan antara label sebenarnya dan hasil prediksi model.
    
    - **Baris** = label asli (ground truth)
    - **Kolom** = label hasil prediksi
    - Angka di **diagonal utama** adalah jumlah prediksi yang benar.
    - Angka di luar diagonal = prediksi yang salah.

    📝 **Contoh pada hasil:**
    - `Low → Low` = 297 ✅
    - `Moderate → Moderate` = 674 ✅
    - `High → High` = 1029 ✅
    - Tidak ada angka di luar diagonal → tidak ada kesalahan prediksi.

    ✅ Ini menunjukkan model **sangat akurat** dalam memetakan data ke kelas stres yang benar.
    """)

    st.subheader("📉 ROC Curve")
    y_bin = label_binarize(y, classes=[0, 1, 2])
    fig_roc, ax = plt.subplots()
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)
    st.markdown("""
    **ROC Curve** (Receiver Operating Characteristic) menunjukkan hubungan antara:
    
    - **True Positive Rate (Recall)** = Seberapa banyak kasus positif yang terdeteksi dengan benar.
    - **False Positive Rate** = Seberapa banyak kasus negatif yang salah dikira positif.

    Garis ROC yang bagus akan **mendekati pojok kiri atas**.

    **AUC (Area Under Curve)** mengukur luas area di bawah kurva ROC.
    - Nilai AUC = 1.0 artinya sempurna.
    - Nilai AUC = 0.5 artinya sama seperti menebak secara acak.

    📝 **Contoh hasil:**
    - AUC untuk semua kelas (`Low`, `Moderate`, `High`) = **1.00**
    - Artinya model sangat hebat dalam membedakan ketiga tingkat stres.
    """)


    st.subheader("🧾 Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y, y_pred, target_names=class_labels, output_dict=True)).T)
    st.markdown("""
    **Classification Report** memberikan ringkasan metrik evaluasi untuk setiap kelas:

    - **Precision**: Dari semua prediksi ke kelas ini, berapa yang benar.
    - **Recall**: Dari semua data yang sebenarnya milik kelas ini, berapa yang berhasil ditemukan.
    - **F1-Score**: Rata-rata harmonis dari precision dan recall.
    - **Support**: Jumlah data asli di kelas tersebut.

    📝 **Contoh Interpretasi:**
    - Jika `High` punya precision dan recall = 1.00 → model memprediksi kelas ini **dengan sempurna**.
    - `Support` menunjukkan distribusi data asli, contohnya:
      - `Low`: 297 mahasiswa
      - `Moderate`: 674 mahasiswa
      - `High`: 1029 mahasiswa
    """)

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
    - Penyeimbangan kelas target menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**  
      > Teknik ini efektif mengatasi dominasi label tertentu (misal `High`) agar model tidak bias.
    
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
# 6. Prediksi
# ===========================
elif page == "Prediksi":
    st.title("🔮 Prediksi Tingkat Stres Mahasiswa")

    # 👤 Input identitas di sini (bukan halaman terpisah)
    nama = st.text_input("Nama Anda:")
    umur = st.number_input("Umur Anda:", min_value=5, max_value=100, value=20)

    # 🧾 Input fitur prediksi
    study = st.slider("Jam Belajar per Hari", 0, 12, 4)
    sleep = st.slider("Jam Tidur per Hari", 0, 12, 7)
    activity = st.slider("Aktivitas Fisik per Hari", 0, 5, 2)
    social = st.slider("Jam Sosialisasi per Hari", 0, 6, 2)
    extracurricular = st.selectbox("Ikut Ekstrakurikuler?", ["Ya", "Tidak"])
    gpa = st.number_input("GPA", 0.0, 4.0, 3.2)

    academic_perf = 'Excellent' if gpa >= 3.5 else 'Good' if gpa >= 3.0 else 'Fair' if gpa >= 2.0 else 'Poor'
    performance_encoded = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}[academic_perf]

    input_df = pd.DataFrame([{
        "Study_Hours_Per_Day": study,
        "Sleep_Hours_Per_Day": sleep,
        "Physical_Activity_Hours_Per_Day": activity,
        "Social_Hours_Per_Day": social,
        "Extracurricular_Hours_Per_Day": 1 if extracurricular == "Ya" else 0,
        "GPA": gpa,
        "Academic_Performance_Encoded": performance_encoded
    }])[features]

    input_df = pd.DataFrame(input_df, columns=scaler.feature_names_in_)
    input_scaled = scaler.transform(input_df)

    if st.button("Prediksi"):
        if nama.strip() == "":
            st.error("❌ Mohon isi nama terlebih dahulu sebelum melakukan prediksi.")
        else:
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            label = {0: "Low", 1: "Moderate", 2: "High"}[pred]
    
            st.success(f"Hai {nama} (umur {umur}), tingkat stresmu diprediksi: **{label}**")
    
            st.subheader("📊 Probabilitas Prediksi")
            fig, ax = plt.subplots()
            ax.bar(["Low", "Moderate", "High"], prob, color=["green", "orange", "red"])
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

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
