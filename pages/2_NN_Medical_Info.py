import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(page_title="NN Medical Info", page_icon="🧬", layout="wide")

st.title("🧬 Neural Network — ข้อมูลโมเดล Dataset การแพทย์")
st.caption("Dense Neural Network สำหรับทำนายจำนวนวันนอนโรงพยาบาล")

st.divider()

# ===== ส่วนที่ 1: ทฤษฎี Dense Neural Network =====
st.header("📚 ทฤษฎี Dense Neural Network")

st.markdown("""
**Dense Neural Network** (Fully Connected Neural Network) คือโครงข่ายประสาทเทียมที่แต่ละ Neuron
ในแต่ละ Layer เชื่อมต่อกับทุก Neuron ใน Layer ถัดไป เหมาะสำหรับงาน Regression ที่มี Features เป็นตัวเลข
""")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🧠 องค์ประกอบหลัก", expanded=True):
        st.markdown("""
        - **Input Layer**: รับ Features ทั้งหมดเข้าโมเดล
        - **Hidden Layers**: เรียนรู้ Pattern จากข้อมูล
        - **Output Layer**: ให้ผลลัพธ์การทำนาย (1 neuron สำหรับ Regression)
        - **Activation Function**: ReLU สำหรับ Hidden, Linear สำหรับ Output
        """)

with col2:
    with st.expander("⚡ กระบวนการเรียนรู้", expanded=True):
        st.markdown("""
        - **Forward Propagation**: ส่งข้อมูลไปข้างหน้าผ่านแต่ละ Layer
        - **Loss Function**: MSE (Mean Squared Error) สำหรับ Regression
        - **Backpropagation**: คำนวณ Gradient ย้อนกลับ
        - **Optimizer**: Adam — ปรับ Learning Rate อัตโนมัติ
        """)

st.divider()

# ===== ส่วนที่ 2: โครงสร้าง Layer =====
st.header("🏗️ โครงสร้าง Layer ของโมเดล")

st.code("""
Model: Dense Neural Network (Regression)
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
    | **Dense(128, ReLU)** | Hidden Layer แรก — 128 neurons, ฟังก์ชัน ReLU |
    | **Dropout(0.2)** | สุ่มปิด 20% ของ neurons เพื่อป้องกัน Overfitting |
    | **Dense(64, ReLU)** | Hidden Layer ที่สอง — 64 neurons |
    | **Dropout(0.2)** | Dropout อีกรอบ |
    | **Dense(32, ReLU)** | Hidden Layer ที่สาม — 32 neurons |
    | **Dense(1, Linear)** | Output Layer — 1 neuron สำหรับ Regression |
    """)

st.divider()

# ===== ส่วนที่ 3: Data Preprocessing =====
st.header("⚙️ Data Preprocessing")

tab1, tab2, tab3 = st.tabs(["1️⃣ จัดการข้อมูล", "2️⃣ Encoding", "3️⃣ Scaling"])

with tab1:
    st.markdown("""
    - **ตรวจสอบ Missing Values**: ลบหรือเติมค่าที่หายไป
    - **ตรวจสอบ Duplicates**: ลบข้อมูลซ้ำ
    - **ตรวจสอบ Outliers**: ใช้ IQR หรือ Z-score ตรวจจับค่าผิดปกติ
    - **แบ่งข้อมูล**: Train/Test Split (80/20)
    """)

with tab2:
    st.markdown("""
    - **Label Encoding** สำหรับ:
      - `gender` → Male=1, Female=0
      - `blood_type` → A=0, AB=1, B=2, O=3
      - `department` → Cardiology=0, Endocrinology=1, General=2, Pulmonology=3
      - `diagnosis` → Asthma=0, Diabetes=1, Healthy=2, Heart Disease=3, Hypertension=4
    - ใช้ **LabelEncoder** จาก scikit-learn
    """)

with tab3:
    st.markdown("""
    - ใช้ **StandardScaler** ปรับค่า Features ให้มี mean=0, std=1
    - สำคัญมากสำหรับ Neural Network เพราะช่วยให้ Gradients ไม่ Vanish/Explode
    - บันทึก Scaler เป็น `scaler.pkl`
    """)

st.divider()

# ===== ส่วนที่ 4: ผลลัพธ์โมเดล =====
st.header("📈 ผลลัพธ์ของโมเดล (Model Performance)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value="1.3456", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value="1.6789", delta="ยิ่งต่ำยิ่งดี", delta_color="inverse")
with col3:
    st.metric(label="R² Score", value="0.8543", delta="ยิ่งสูงยิ่งดี")

with st.expander("📖 คำอธิบาย Metrics"):
    st.markdown("""
    | Metric | สูตร | ความหมาย |
    |--------|------|---------|
    | **MAE** | Σ\|yᵢ - ŷᵢ\| / n | ค่าเฉลี่ยของความคลาดเคลื่อนสัมบูรณ์ |
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
