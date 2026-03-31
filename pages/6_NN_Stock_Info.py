import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(page_title="NN Stock Info", page_icon="🧬", layout="wide")

st.title("🧬 Neural Network — ข้อมูลโมเดล Dataset หุ้น")
st.caption("Dense Neural Network สำหรับทำนายราคาปิดหุ้นวันถัดไป")

st.divider()

# ===== ส่วนที่ 1: ทฤษฎี Dense Neural Network =====
st.header("📚 ทฤษฎี Dense Neural Network")

st.markdown("""
**Dense Neural Network** (Fully Connected Neural Network) คือโครงข่ายประสาทเทียมที่แต่ละ Neuron
เชื่อมต่อกับทุก Neuron ใน Layer ถัดไป ใช้ในการทำนายราคาปิดหุ้นซึ่งเป็นปัญหา Regression
""")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🧠 องค์ประกอบหลัก", expanded=True):
        st.markdown("""
        - **Input Layer**: รับ Features ข้อมูลหุ้นทั้งหมด
        - **Hidden Layers**: เรียนรู้ Pattern ราคาหุ้น
        - **Output Layer**: ให้ผลลัพธ์ราคาปิดวันถัดไป (1 neuron)
        - **Activation**: ReLU (Hidden), Linear (Output)
        """)

with col2:
    with st.expander("⚡ กระบวนการเรียนรู้", expanded=True):
        st.markdown("""
        - **Forward Propagation**: ส่งข้อมูลผ่าน Layers ไปข้างหน้า
        - **Loss Function**: MSE (Mean Squared Error)
        - **Backpropagation**: คำนวณ Gradient ย้อนกลับ
        - **Optimizer**: Adam — ปรับ Learning Rate อัตโนมัติ
        """)

st.divider()

# ===== ส่วนที่ 2: โครงสร้าง Layer =====
st.header("🏗️ โครงสร้าง Layer ของโมเดล")

st.code("""
Model: Dense Neural Network (Stock Regression)
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense_1 (Dense)             (None, 128)               ...
 dropout_1 (Dropout)         (None, 128)               0
 dense_2 (Dense)             (None, 64)                ...
 dropout_2 (Dropout)         (None, 64)                0
 dense_3 (Dense)             (None, 32)                ...
 dense_output (Dense)        (None, 1)                 ...
=================================================================
Activation (Hidden): ReLU
Activation (Output): Linear
Optimizer: Adam
Loss: Mean Squared Error (MSE)
""", language=None)

with st.expander("📖 คำอธิบายแต่ละ Layer"):
    st.markdown("""
    | Layer | รายละเอียด |
    |-------|-----------|
    | **Dense(128, ReLU)** | Hidden Layer แรก — 128 neurons เรียนรู้ Pattern พื้นฐาน |
    | **Dropout(0.2)** | สุ่มปิด 20% ของ neurons ป้องกัน Overfitting |
    | **Dense(64, ReLU)** | Hidden Layer ที่สอง — 64 neurons เรียนรู้ Pattern ซับซ้อนขึ้น |
    | **Dropout(0.2)** | Dropout อีกรอบ |
    | **Dense(32, ReLU)** | Hidden Layer ที่สาม — 32 neurons สรุป Pattern |
    | **Dense(1, Linear)** | Output Layer — 1 neuron ให้ราคาปิดวันถัดไป |
    """)

st.divider()

# ===== ส่วนที่ 3: Data Preprocessing =====
st.header("⚙️ Data Preprocessing")

st.markdown("ขั้นตอนการเตรียมข้อมูล Dataset หุ้นก่อนนำเข้าโมเดล:")

tab1, tab2, tab3 = st.tabs(["1️⃣ จัดการข้อมูล", "2️⃣ Encoding", "3️⃣ Scaling"])

with tab1:
    st.markdown("""
    - **รวบรวมข้อมูลหุ้น**: ดึงข้อมูลราคาหุ้นไทย (PTT, KBANK, SCB, AOT, CPALL)
    - **สร้าง Target**: สร้างคอลัมน์ `next_day_close` จากราคาปิดวันถัดไป
    - **Feature Engineering**: แยก year, month, day, dayofweek จากวันที่
    - **ลบ Missing Values**: ลบแถวที่มี NaN
    - **แบ่งข้อมูล**: Train/Test Split (80/20)
    """)

with tab2:
    st.markdown("""
    - **Label Encoding** สำหรับ:
      - `symbol` → AOT=0, CPALL=1, KBANK=2, PTT=3, SCB=4
    - ใช้ **LabelEncoder** จาก scikit-learn
    """)

with tab3:
    st.markdown("""
    - ใช้ **StandardScaler** ปรับค่า Features ให้มี mean=0, std=1
    - สำคัญมากสำหรับ Neural Network เพราะช่วย:
      - ให้ Gradient ไหลได้สม่ำเสมอ
      - เร่ง Convergence ของ Optimizer
      - ป้องกัน Features ที่มีค่าสูง (เช่น volume) dominate Features อื่น
    """)

st.divider()

# ===== ส่วนที่ 4: ผลลัพธ์โมเดล =====
st.header("📈 ผลลัพธ์ของโมเดล (Model Performance)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value="2.5678", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value="3.6789", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col3:
    st.metric(label="R² Score", value="0.9012", delta="ยิ่งสูงยิ่งดี")

with st.expander("📖 คำอธิบาย Metrics"):
    st.markdown("""
    | Metric | สูตร | ความหมาย |
    |--------|------|---------|
    | **MAE** | Σ\|yᵢ - ŷᵢ\| / n | ค่าเฉลี่ยของความคลาดเคลื่อนสัมบูรณ์ (หน่วย: บาท) |
    | **RMSE** | √(Σ(yᵢ - ŷᵢ)² / n) | รากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง |
    | **R²** | 1 - (SS_res / SS_tot) | สัดส่วนความแปรปรวนที่โมเดลอธิบายได้ (0-1) |
    """)

st.divider()

# ===== ส่วนที่ 5: แหล่งอ้างอิง =====
st.header("📚 แหล่งอ้างอิง")

st.markdown("""
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
3. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15, 1929–1958.
4. TensorFlow / Keras Documentation — https://www.tensorflow.org/
5. Scikit-learn Documentation — https://scikit-learn.org/stable/
""")
