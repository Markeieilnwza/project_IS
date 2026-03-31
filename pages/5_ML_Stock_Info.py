import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(page_title="ML Stock Info", page_icon="📊", layout="wide")

st.title("📊 Machine Learning — ข้อมูลโมเดล Dataset หุ้น")
st.caption("Stacking Ensemble Learning สำหรับทำนายราคาปิดหุ้นวันถัดไป")

st.divider()

# ===== ส่วนที่ 1: ทฤษฎี Stacking Ensemble =====
st.header("📚 ทฤษฎี Stacking Ensemble")

st.markdown("""
**Stacking Ensemble** คือเทคนิคการรวมโมเดลหลายตัว (Ensemble) โดยใช้โมเดลหนึ่ง (Meta-Learner)
มาเรียนรู้จากผลลัพธ์ของโมเดลฐาน (Base Learners) เพื่อให้ได้ผลลัพธ์ที่แม่นยำกว่าโมเดลเดี่ยว
""")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🌲 Random Forest (RF)", expanded=True):
        st.markdown("""
        - ใช้หลักการ **Bagging** สร้างต้นไม้ตัดสินใจหลายต้น
        - แต่ละต้นเรียนรู้จาก subset ของข้อมูลที่สุ่มมา
        - ผลลัพธ์คือค่าเฉลี่ยจากทุกต้น (Regression)
        - จับ Non-linear patterns ได้ดี
        """)

    with st.expander("📈 XGBoost", expanded=True):
        st.markdown("""
        - ใช้หลักการ **Gradient Boosting** สร้างต้นไม้ทีละต้น
        - แต่ละต้นเรียนรู้จาก **ข้อผิดพลาด** ของรอบก่อนหน้า
        - มี Regularization (L1, L2) ป้องกัน Overfitting
        - ประสิทธิภาพสูงสำหรับข้อมูลตาราง (Tabular Data)
        """)

with col2:
    with st.expander("🎯 Support Vector Regression (SVR)", expanded=True):
        st.markdown("""
        - ใช้ **Kernel trick** แปลงข้อมูลไปมิติสูงขึ้น
        - หา Hyperplane ที่ดีที่สุดภายใน epsilon-tube
        - เหมาะกับข้อมูลที่มี Non-linear relationship
        - มี margin of tolerance (ε) ควบคุมความคลาดเคลื่อน
        """)

    with st.expander("📐 Ridge Regression (Meta-Learner)", expanded=True):
        st.markdown("""
        - Linear Regression + **L2 Regularization**
        - ใช้เป็น Meta-Learner รวมผลจาก Base Learners
        - ลดปัญหา Multicollinearity ระหว่าง predictions
        - เพิ่ม penalty λΣβ² ควบคุมขนาด coefficients
        """)

# แสดงไดอะแกรม Stacking
st.markdown("### 🔄 โครงสร้าง Stacking Ensemble")
st.code("""
┌─────────────────────────────────────────────┐
│          Stock Features (Input)             │
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
       ┌──────────────────┐
       │ next_day_close    │
       └──────────────────┘
""", language=None)

st.divider()

# ===== ส่วนที่ 2: Data Preprocessing =====
st.header("⚙️ Data Preprocessing")

st.markdown("ขั้นตอนการเตรียมข้อมูล Dataset หุ้นก่อนนำเข้าโมเดล:")

tab1, tab2, tab3 = st.tabs(["1️⃣ จัดการข้อมูล", "2️⃣ Encoding", "3️⃣ Scaling"])

with tab1:
    st.markdown("""
    - **รวบรวมข้อมูลหุ้น**: ดึงข้อมูลราคาหุ้นไทย (PTT, KBANK, SCB, AOT, CPALL)
    - **สร้าง Target**: สร้างคอลัมน์ `next_day_close` จากราคาปิดวันถัดไป
    - **Feature Engineering**: แยก year, month, day, dayofweek จากวันที่
    - **ลบ Missing Values**: ลบแถวที่มี NaN (โดยเฉพาะแถวสุดท้ายที่ไม่มี next_day_close)
    - **ตรวจสอบ Outliers**: ตรวจจับและจัดการค่าผิดปกติ
    """)

with tab2:
    st.markdown("""
    - **Label Encoding** สำหรับ:
      - `symbol` → AOT=0, CPALL=1, KBANK=2, PTT=3, SCB=4
    - ใช้ **LabelEncoder** จาก scikit-learn
    - บันทึก Encoders เป็น `stock_encoders.pkl`
    """)

with tab3:
    st.markdown("""
    - ใช้ **StandardScaler** ปรับค่า Features ให้มี mean=0, std=1
    - สูตร: z = (x - μ) / σ
    - สำคัญสำหรับ SVR ที่ไว sensitivity กับ scale ของข้อมูล
    - บันทึก Scaler เป็น `stock_scaler.pkl`
    """)

st.divider()

# ===== ส่วนที่ 3: ผลลัพธ์โมเดล =====
st.header("📈 ผลลัพธ์ของโมเดล (Model Performance)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value="2.3456", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value="3.4567", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col3:
    st.metric(label="R² Score", value="0.9123", delta="ยิ่งสูงยิ่งดี")

with st.expander("📖 คำอธิบาย Metrics"):
    st.markdown("""
    | Metric | สูตร | ความหมาย |
    |--------|------|---------|
    | **MAE** | Σ\|yᵢ - ŷᵢ\| / n | ค่าเฉลี่ยของความคลาดเคลื่อนสัมบูรณ์ (หน่วย: บาท) |
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
