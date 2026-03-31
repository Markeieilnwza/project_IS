import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ตั้งค่าหน้า
st.set_page_config(page_title="NN Medical Test", page_icon="🧪", layout="wide")

st.title("🧪 NN Medical Test — ทำนายจำนวนวันนอนโรงพยาบาล")
st.caption("ใช้ Dense Neural Network ทำนายผล")

st.divider()

# ===== โหลดโมเดลและ Preprocessors =====
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "medel", "med")

@st.cache_resource
def load_model():
    """โหลดโมเดล Neural Network, Scaler และ Encoders"""
    from tensorflow.keras.models import load_model as keras_load
    model = keras_load(os.path.join(MODEL_DIR, "neural_network_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    return model, scaler, encoders

# พยายามโหลดโมเดล
try:
    model, scaler, encoders = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")

st.divider()

# ===== ฟอร์มรับ Input =====
st.header("📝 กรอกข้อมูลผู้ป่วย")

with st.form("nn_medical_form"):
    # แถวที่ 1: ข้อมูลพื้นฐาน
    st.subheader("👤 ข้อมูลพื้นฐาน")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("อายุ (Age)", min_value=0, max_value=120, value=45, step=1)
    with col2:
        gender = st.selectbox("เพศ (Gender)", options=["Male", "Female"])
    with col3:
        blood_type = st.selectbox("กรุ๊ปเลือด (Blood Type)", options=["A", "B", "AB", "O"])

    # แถวที่ 2: ข้อมูลร่างกาย
    st.subheader("📏 ข้อมูลร่างกาย")
    col4, col5, col6 = st.columns(3)

    with col4:
        weight_kg = st.number_input("น้ำหนัก (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    with col5:
        height_cm = st.number_input("ส่วนสูง (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    with col6:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.2, step=0.1)

    # แถวที่ 3: ข้อมูลสุขภาพ
    st.subheader("❤️ สัญญาณชีพ")
    col7, col8, col9, col10 = st.columns(4)

    with col7:
        systolic_bp = st.number_input("ความดัน Systolic (mmHg)", min_value=60, max_value=250, value=120, step=1)
    with col8:
        diastolic_bp = st.number_input("ความดัน Diastolic (mmHg)", min_value=40, max_value=150, value=80, step=1)
    with col9:
        glucose_mg_dl = st.number_input("น้ำตาล (mg/dL)", min_value=50.0, max_value=500.0, value=100.0, step=0.1)
    with col10:
        cholesterol_mg_dl = st.number_input("คอเลสเตอรอล (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=0.1)

    # แถวที่ 4: อัตราการเต้นหัวใจ พฤติกรรม
    st.subheader("🏃 พฤติกรรมสุขภาพ")
    col11, col12, col13 = st.columns(3)

    with col11:
        heart_rate = st.number_input("อัตราการเต้นหัวใจ (bpm)", min_value=40, max_value=200, value=72, step=1)
    with col12:
        smoker = st.selectbox("สูบบุหรี่ (Smoker)", options=[0, 1], format_func=lambda x: "ไม่สูบ" if x == 0 else "สูบ")
    with col13:
        exercise_days_week = st.slider("จำนวนวันออกกำลังกาย/สัปดาห์", min_value=0, max_value=7, value=3)

    # แถวที่ 5: แผนกและการวินิจฉัย
    st.subheader("🏥 ข้อมูลทางการแพทย์")
    col14, col15 = st.columns(2)

    with col14:
        department = st.selectbox("แผนก (Department)",
                                  options=["Cardiology", "Endocrinology", "General", "Pulmonology"])
    with col15:
        diagnosis = st.selectbox("การวินิจฉัย (Diagnosis)",
                                 options=["Diabetes", "Hypertension", "Heart Disease", "Healthy", "Asthma"])

    # ปุ่มทำนาย
    submitted = st.form_submit_button("🔮 ทำนายจำนวนวันนอนโรงพยาบาล", use_container_width=True)

# ===== ประมวลผลและแสดงผลลัพธ์ =====
if submitted:
    if not model_loaded:
        st.error("❌ โมเดลยังไม่ถูกโหลด กรุณาตรวจสอบไฟล์โมเดล")
    else:
        # เตรียมข้อมูล input
        input_data = {
            "age": age,
            "gender": gender,
            "blood_type": blood_type,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose_mg_dl": glucose_mg_dl,
            "cholesterol_mg_dl": cholesterol_mg_dl,
            "heart_rate": heart_rate,
            "smoker": smoker,
            "exercise_days_week": exercise_days_week,
            "department": department,
            "diagnosis": diagnosis
        }

        # สร้าง DataFrame
        df_input = pd.DataFrame([input_data])

        # Encode categorical features ด้วย encoders ที่บันทึกไว้
        categorical_cols = ["gender", "blood_type", "department", "diagnosis"]
        for col in categorical_cols:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        # Scale features ด้วย scaler ที่บันทึกไว้
        df_scaled = scaler.transform(df_input)

        # ทำนายด้วย Neural Network
        prediction = model.predict(df_scaled)
        predicted_days = round(float(prediction[0][0]), 2)

        st.divider()

        # แสดงผลลัพธ์
        st.header("🎯 ผลการทำนาย")
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r2:
            st.metric(
                label="จำนวนวันนอนโรงพยาบาล (Hospital Stay Days)",
                value=f"{predicted_days} วัน"
            )
            st.success(f"✅ ผลการทำนาย: ผู้ป่วยคาดว่าจะนอนโรงพยาบาลประมาณ **{predicted_days} วัน**")

        # แสดงสรุป input ที่กรอก
        with st.expander("📋 สรุปข้อมูลที่กรอก"):
            st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
