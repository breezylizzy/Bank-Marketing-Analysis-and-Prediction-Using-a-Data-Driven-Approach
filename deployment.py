import streamlit as st
import pandas as pd
import joblib
import sklearn

model = joblib.load("stacking_model.pkl")

st.title("💰 Prediksi Deposit Nasabah (Bank Marketing)")
st.markdown("Masukkan data nasabah di bawah ini:")

age = st.number_input("Umur", 18, 100, 35)
job = st.selectbox("Pekerjaan", [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
])
marital = st.selectbox("Status Pernikahan", ['married', 'single', 'divorced'])
education = st.selectbox("Pendidikan", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Kredit macet (default)?", ['yes', 'no', 'unknown'])
balance = st.number_input("Saldo (balance)", value=1000)
housing = st.selectbox("Punya pinjaman rumah?", ['yes', 'no', 'unknown'])
loan = st.selectbox("Punya pinjaman personal?", ['yes', 'no', 'unknown'])
contact = st.selectbox("Jenis kontak", ['cellular', 'telephone', 'unknown'])
day = st.number_input("Hari terakhir kontak", 1, 31, 15)
month = st.selectbox("Bulan kontak", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
campaign = st.number_input("Jumlah kontak selama kampanye", 1, 50, 1)
pdays = st.number_input("Hari sejak kontak sebelumnya", -1, 999, 999)
previous = st.number_input("Jumlah kontak sebelumnya", 0, 50, 0)
poutcome = st.selectbox("Hasil kontak sebelumnya", ['success', 'failure', 'other', 'unknown'])

input_data = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}])

st.write("📋 Data Input:")
st.dataframe(input_data)

def add_features(df):
    df = df.copy()
    df["contacts_per_prev"] =  df['campaign'] / (df['previous'] + 1).replace(0, 1)
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["negative_balance"] = (df["balance"] < 0).astype(int)
    df["has_any_loan"] = ((df["loan"] == "yes") | (df["housing"] == "yes")).astype(int)
    df["prev_success_flag"] = (df["poutcome"] == "success").astype(int)
    df["contact_efficiency"] = df['previous'] / (df['campaign'] + 1).replace(0, 1)
    df["age_group"] = pd.cut(df["age"], bins=[14, 24, 34, 49, 64, 100],
                             labels=['Young Adult', 'Early Working Adult', 'Middle-aged Adult', 'Mature Adult', 'Senior / Elderly'])
    return df

input_data_fe = add_features(input_data)

if st.button("🚀 Prediksi"):
    try:
        prob = model.predict_proba(input_data_fe)[0, 1]
        pred = int(prob >= 0.3)

        st.subheader("🔮 Hasil Prediksi")

        if pred == 1:
            st.success(f"✅ Nasabah **berpotensi melakukan DEPOSIT** (Probabilitas: {prob:.2f})")
        else:
            st.error(f"❌ Nasabah **tidak berpotensi deposit** (Probabilitas: {prob:.2f})")

        st.progress(float(prob))
    except Exception as e:
        st.error(f"Terjadi error: {e}")
