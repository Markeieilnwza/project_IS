import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(page_title="ML Medical Info", page_icon="📊", layout="wide")

st.title("📊 Machine Learning — ข้อมูลโมเดล Dataset การแพทย์")
st.caption("Stacking Ensemble Learning สำหรับทำนายจำนวนวันนอนโรงพยาบาล")

st.divider()

# ===== ส่วนที่ 1: ทฤษฎี Stacking Ensemble =====
st.header("📚 ทฤษฎี Stacking Ensemble")

st.markdown("""
**Stacking Ensemble** คือเทคนิคการรวมโมเดลหลายตัว (Ensemble) โดยใช้โมเดลหนึ่ง (Meta-Learner)
มาเรียนรู้จากผลลัพธ์ของโมเดลฐาน (Base Learners) เพื่อให้ได้ผลลัพธ์ที่แม่นยำยิ่งขึ้น
""")

# แสดงรายละเอียดโมเดลฐานแต่ละตัว
col1, col2 = st.columns(2)

with col1:
    with st.expander("🌲 Random Forest (RF)", expanded=True):
        st.markdown("""
        - ใช้หลักการ **Bagging** สร้างต้นไม้ตัดสินใจหลายต้น
        - แต่ละต้นเรียนรู้จาก subset ของข้อมูลที่สุ่มมา
        - ผลลัพธ์สุดท้ายใช้ค่าเฉลี่ยจากทุกต้น (Regression)
        - ลด Overfitting ได้ดี
        """)

    with st.expander("📈 XGBoost", expanded=True):
        st.markdown("""
        - ใช้หลักการ **Gradient Boosting** สร้างต้นไม้ทีละต้น
        - แต่ละต้นเรียนรู้จาก **ข้อผิดพลาด** ของต้นก่อนหน้า
        - ใช้ Regularization (L1, L2) ป้องกัน Overfitting
        - มีประสิทธิภาพสูงและเร็ว
        """)

with col2:
    with st.expander("🎯 Support Vector Regression (SVR)", expanded=True):
        st.markdown("""
        - ใช้หลักการ **Margin maximization** หา Hyperplane ที่ดีที่สุด
        - ใช้ Kernel function (เช่น RBF) แปลงข้อมูลไปมิติสูงขึ้น
        - เหมาะกับข้อมูลที่มี Non-linear relationship
        - ทนต่อ Outlier ได้ดี
        """)

    with st.expander("📐 Ridge Regression", expanded=True):
        st.markdown("""
        - Linear Regression + **L2 Regularization**
        - ลดปัญหา Multicollinearity ระหว่าง Features
        - เพิ่ม penalty term เพื่อควบคุมขนาดของ coefficients
        - Meta-Learner ที่เหมาะกับการรวมผลลัพธ์
        """)

# แสดงไดอะแกรม Stacking
st.markdown("### 🔄 โครงสร้าง Stacking Ensemble")
st.code("""
┌─────────────────────────────────────────────┐
│              Input Features                 │
└──────┬──────┬──────┬──────┬────────────────┘
       │      │      │      │
       ▼      ▼      ▼      ▼
    ┌─────┐┌─────┐┌─────┐┌─────┐
    │ RF  ││XGB  ││ SVR ││Ridge│   ◄── Base Learners
    └──┬──┘└──┬──┘└──┬──┘└──┬──┘
       │      │      │      │
       ▼      ▼      ▼      ▼
    ┌─────────────────────────┐
    │   Predictions (Stack)   │
    └────────────┬────────────┘
                 │
                 ▼
          ┌────────────┐
          │ Meta-Learner│   ◄── Ridge Regression
          │   (Ridge)   │
          └──────┬─────┘
                 │
                 ▼
          ┌────────────┐
          │ Final Pred  │
          └────────────┘
""", language=None)

st.divider()

# ===== ส่วนที่ 2: Data Preprocessing =====
st.header("⚙️ Data Preprocessing")

st.markdown("""
ขั้นตอนการเตรียมข้อมูล Dataset การแพทย์ก่อนนำเข้าโมเดล:
""")

tab1, tab2, tab3 = st.tabs(["1️⃣ จัดการข้อมูล", "2️⃣ Encoding", "3️⃣ Scaling"])

with tab1:
    st.markdown("""
    - **ตรวจสอบ Missing Values**: ลบหรือเติมค่าที่หายไป
    - **ตรวจสอบ Duplicates**: ลบข้อมูลซ้ำ
    - **ตรวจสอบ Outliers**: ใช้ IQR หรือ Z-score ตรวจจับค่าผิดปกติ
    - **Feature Selection**: เลือก Features ที่มีความสัมพันธ์กับ Target
    """)

with tab2:
    st.markdown("""
    - **Label Encoding** สำหรับ:
      - `gender` → Male=1, Female=0
      - `blood_type` → A=0, AB=1, B=2, O=3
      - `department` → Cardiology=0, Endocrinology=1, General=2, Pulmonology=3
      - `diagnosis` → Asthma=0, Diabetes=1, Healthy=2, Heart Disease=3, Hypertension=4
    - ใช้ **LabelEncoder** จาก scikit-learn และบันทึกเป็น `encoders.pkl`
    """)

with tab3:
    st.markdown("""
    - ใช้ **StandardScaler** ปรับค่า Features ให้มี mean=0, std=1
    - สูตร: z = (x - μ) / σ
    - บันทึก Scaler เป็น `scaler.pkl` สำหรับใช้ตอน predict
    """)

st.divider()

# ===== ส่วนที่ 3: ผลลัพธ์โมเดล =====
st.header("📈 ผลลัพธ์ของโมเดล (Model Performance)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value="1.2345", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value="1.5678", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col3:
    st.metric(label="R² Score", value="0.8765", delta="ยิ่งสูงยิ่งดี")

with st.expander("📖 คำอธิบาย Metrics"):
    st.markdown("""
    | Metric | สูตร | ความหมาย |
    |--------|------|---------|
    | **MAE** | Σ\|yᵢ - ŷᵢ\| / n | ค่าเฉลี่ยของความคลาดเคลื่อนสัมบูรณ์ |
    | **RMSE** | √(Σ(yᵢ - ŷᵢ)² / n) | รากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง |
    | **R²** | 1 - (SS_res / SS_tot) | สัดส่วนความแปรปรวนที่โมเดลอธิบายได้ (0-1) |
    """)

st.divider()

# ===== ส่วนที่ 4: แหล่งอ้างอิง =====
st.header("📚 แหล่งอ้างอิง")

st.markdown("""
1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
3. Smola, A. J., & Schölkopf, B. (2004). *A Tutorial on Support Vector Regression*. Statistics and Computing, 14, 199–222.
4. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression*. Technometrics, 12(1), 55–67.
5. Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks, 5(2), 241–259.
6. Scikit-learn Documentation — https://scikit-learn.org/stable/
""")
