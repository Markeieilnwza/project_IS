import streamlit as st

# ตั้งค่าหน้าเว็บหลัก
st.set_page_config(
    page_title="ML & Neural Network Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ส่วนหัวของหน้าแรก =====
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem;">🧠 Machine Learning & Neural Network</h1>
        <h2 style="font-size: 1.5rem; color: gray;">Prediction Web Application</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ===== คำอธิบายโปรเจค =====
st.markdown("## 📌 เกี่ยวกับโปรเจคนี้")
st.info(
    "เว็บแอปพลิเคชันนี้เป็นโปรเจค Machine Learning ที่ใช้ **2 Dataset** ได้แก่ "
    "**Dataset การแพทย์** (ทำนายจำนวนวันนอนโรงพยาบาล) และ "
    "**Dataset หุ้น** (ทำนายราคาปิดวันถัดไป) "
    "โดยใช้ 2 วิธีคือ **Stacking Ensemble** และ **Dense Neural Network**"
)

st.divider()

# ===== เมนูนำทาง 8 หน้า =====
st.markdown("## 🗂️ เมนูนำทาง")

# --- Dataset การแพทย์ ---
st.markdown("### 🏥 Dataset การแพทย์ — ทำนายจำนวนวันนอนโรงพยาบาล")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_ML_Medical_Info.py", label="📊 ML Medical Info", icon="1️⃣")
    st.caption("ทฤษฎี Stacking Ensemble สำหรับ Dataset การแพทย์")

with col2:
    st.page_link("pages/2_NN_Medical_Info.py", label="🧬 NN Medical Info", icon="2️⃣")
    st.caption("ทฤษฎี Dense Neural Network สำหรับ Dataset การแพทย์")

with col3:
    st.page_link("pages/3_ML_Medical_Test.py", label="🔬 ML Medical Test", icon="3️⃣")
    st.caption("ทดสอบทำนายด้วย Stacking Ensemble")

with col4:
    st.page_link("pages/4_NN_Medical_Test.py", label="🧪 NN Medical Test", icon="4️⃣")
    st.caption("ทดสอบทำนายด้วย Neural Network")

st.markdown("---")

# --- Dataset หุ้น ---
st.markdown("### 📈 Dataset หุ้น — ทำนายราคาปิดวันถัดไป")
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.page_link("pages/5_ML_Stock_Info.py", label="📊 ML Stock Info", icon="5️⃣")
    st.caption("ทฤษฎี Stacking Ensemble สำหรับ Dataset หุ้น")

with col6:
    st.page_link("pages/6_NN_Stock_Info.py", label="🧬 NN Stock Info", icon="6️⃣")
    st.caption("ทฤษฎี Dense Neural Network สำหรับ Dataset หุ้น")

with col7:
    st.page_link("pages/7_ML_Stock_Test.py", label="🔬 ML Stock Test", icon="7️⃣")
    st.caption("ทดสอบทำนายด้วย Stacking Ensemble")

with col8:
    st.page_link("pages/8_NN_Stock_Test.py", label="🧪 NN Stock Test", icon="8️⃣")
    st.caption("ทดสอบทำนายด้วย Neural Network")

st.divider()

# ===== ข้อมูลสรุป =====
st.markdown("## 📋 สรุปภาพรวม")

tab1, tab2 = st.tabs(["🏥 การแพทย์", "📈 หุ้น"])

with tab1:
    st.markdown("""
    | รายการ | รายละเอียด |
    |--------|-----------|
    | **Dataset** | ข้อมูลผู้ป่วยจากโรงพยาบาล |
    | **Target** | จำนวนวันนอนโรงพยาบาล (`hospital_stay_days`) |
    | **Features** | อายุ, เพศ, กรุ๊ปเลือด, BMI, ความดัน, น้ำตาล, คอเลสเตอรอล ฯลฯ |
    | **ML Model** | Stacking Ensemble (RF + XGBoost + SVR + Ridge) |
    | **NN Model** | Dense Neural Network |
    """)

with tab2:
    st.markdown("""
    | รายการ | รายละเอียด |
    |--------|-----------|
    | **Dataset** | ข้อมูลราคาหุ้นไทย |
    | **Target** | ราคาปิดวันถัดไป (`next_day_close`) |
    | **Features** | Symbol, Open, High, Low, Close, Volume, PE Ratio ฯลฯ |
    | **ML Model** | Stacking Ensemble (RF + XGBoost + SVR + Ridge) |
    | **NN Model** | Dense Neural Network |
    """)

# ===== Footer =====
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛠️ พัฒนาด้วย Streamlit | Machine Learning & Deep Learning Project"
    "</div>",
    unsafe_allow_html=True
)
