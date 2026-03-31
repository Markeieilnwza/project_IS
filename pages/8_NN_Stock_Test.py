import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ตั้งค่าหน้า
st.set_page_config(page_title="NN Stock Test", page_icon="🧪", layout="wide")

st.title("🧪 NN Stock Test — ทำนายราคาปิดหุ้นวันถัดไป")
st.caption("ใช้ Dense Neural Network ทำนายผล")

st.divider()

# ===== โหลดโมเดลและ Preprocessors =====
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "medel", "stock")

@st.cache_resource
def load_model():
    """โหลดโมเดล Neural Network, Scaler และ Encoders ของหุ้น"""
    from tensorflow.keras.models import load_model as keras_load
    model = keras_load(os.path.join(MODEL_DIR, "stock_neural_network_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "stock_scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "stock_encoders.pkl"))
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
st.header("📝 กรอกข้อมูลหุ้น")

with st.form("stock_nn_form"):
    # แถวที่ 1: สัญลักษณ์หุ้น
    st.subheader("🏷️ สัญลักษณ์หุ้น")
    symbol = st.selectbox("สัญลักษณ์หุ้น (Symbol)", options=["PTT", "KBANK", "SCB", "AOT", "CPALL"])

    # แถวที่ 2: ราคาหุ้น
    st.subheader("💰 ข้อมูลราคา")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        open_price = st.number_input("ราคาเปิด (Open)", min_value=0.0, max_value=10000.0, value=150.0, step=0.01)
    with col2:
        high = st.number_input("ราคาสูงสุด (High)", min_value=0.0, max_value=10000.0, value=155.0, step=0.01)
    with col3:
        low = st.number_input("ราคาต่ำสุด (Low)", min_value=0.0, max_value=10000.0, value=148.0, step=0.01)
    with col4:
        close = st.number_input("ราคาปิด (Close)", min_value=0.0, max_value=10000.0, value=153.0, step=0.01)

    # แถวที่ 3: ปริมาณการซื้อขายและอัตราส่วน
    st.subheader("📊 ข้อมูลการซื้อขาย")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        volume = st.number_input("ปริมาณการซื้อขาย (Volume)", min_value=0, max_value=1000000000, value=5000000, step=1000)
    with col6:
        pe_ratio = st.number_input("PE Ratio", min_value=0.0, max_value=200.0, value=15.0, step=0.01)
    with col7:
        dividend_yield = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.01)
    with col8:
        market_cap_m = st.number_input("Market Cap (ล้านบาท)", min_value=0.0, max_value=5000000.0, value=500000.0, step=100.0)

    # แถวที่ 4: ข้อมูลวันที่
    st.subheader("📅 ข้อมูลวันที่")
    col9, col10, col11, col12 = st.columns(4)

    with col9:
        year = st.number_input("ปี (Year)", min_value=2000, max_value=2030, value=2025, step=1)
    with col10:
        month = st.number_input("เดือน (Month)", min_value=1, max_value=12, value=6, step=1)
    with col11:
        day = st.number_input("วัน (Day)", min_value=1, max_value=31, value=15, step=1)
    with col12:
        dayofweek = st.selectbox("วันในสัปดาห์ (Day of Week)",
                                 options=[0, 1, 2, 3, 4],
                                 format_func=lambda x: ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์"][x])

    # ปุ่มทำนาย
    submitted = st.form_submit_button("🔮 ทำนายราคาปิดวันถัดไป", use_container_width=True)

# ===== ประมวลผลและแสดงผลลัพธ์ =====
if submitted:
    if not model_loaded:
        st.error("❌ โมเดลยังไม่ถูกโหลด กรุณาตรวจสอบไฟล์โมเดล")
    else:
        # เตรียมข้อมูล input
        input_data = {
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield,
            "market_cap_m": market_cap_m,
            "year": year,
            "month": month,
            "day": day,
            "dayofweek": dayofweek
        }

        # สร้าง DataFrame
        df_input = pd.DataFrame([input_data])

        # Encode categorical features ด้วย encoders ที่บันทึกไว้
        categorical_cols = ["symbol"]
        for col in categorical_cols:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        # Scale features ด้วย scaler ที่บันทึกไว้
        df_scaled = scaler.transform(df_input)

        # ทำนายด้วย Neural Network
        prediction = model.predict(df_scaled)
        predicted_price = round(float(prediction[0][0]), 2)

        st.divider()

        # แสดงผลลัพธ์
        st.header("🎯 ผลการทำนาย")
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r2:
            # คำนวณส่วนต่างจากราคาปิดวันนี้
            price_diff = predicted_price - close
            price_diff_pct = (price_diff / close) * 100 if close != 0 else 0

            st.metric(
                label=f"ราคาปิดวันถัดไป ({symbol})",
                value=f"฿{predicted_price:,.2f}",
                delta=f"{price_diff:+.2f} ({price_diff_pct:+.2f}%)"
            )
            st.success(f"✅ ผลการทำนาย: หุ้น **{symbol}** คาดว่าราคาปิดวันถัดไปจะอยู่ที่ **฿{predicted_price:,.2f}**")

        # แสดงสรุป input ที่กรอก
        with st.expander("📋 สรุปข้อมูลที่กรอก"):
            st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
