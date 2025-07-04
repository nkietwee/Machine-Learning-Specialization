# ทางเลือกของ Gradient Descent: Normal Equation

## 🔷 1. Normal Equation คืออะไร?

Normal Equation คือวิธีการหาค่าพารามิเตอร์ของโมเดล **Linear Regression** โดยไม่ต้องใช้การทำซ้ำ (iterations) แบบ Gradient Descent

### สมการของ Linear Regression:

$$
\hat{y} = Xw
$$

- \( X \): เมทริกซ์ของข้อมูล input (รวม bias term ด้วย)  
- \( w \): เวกเตอร์ของพารามิเตอร์ (รวม bias term ด้วย)  
- \( \hat{y} \): ค่าที่โมเดลทำนาย

**เป้าหมาย:** หาค่า \( w \) ที่ทำให้ \( \hat{y} \) ใกล้กับ \( y \) มากที่สุด (ลดค่า cost function)

---

## 🔷 2. วิธีการคำนวณ

### ฟังก์ชัน Cost:

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

หาค่า \( w \) โดยแก้:

$$
\frac{\partial J(w)}{\partial w} = 0
$$

คำตอบคือ:

$$
w = (X^T X)^{-1} X^T y
$$

> 📌 นี่คือ **Normal Equation**

---

## 🔷 3. ตัวอย่าง

### ข้อมูลราคาบ้าน:

| ขนาดบ้าน (ตร.ฟุต) | ราคา (พันบาท) |
|---------------------|----------------|
| 1000                | 200            |
| 1500                | 300            |
| 2000                | 400            |

### เตรียมข้อมูล:

$$
X = \begin{bmatrix}1 & 1000\\ 1 & 1500\\ 1 & 2000\end{bmatrix}, \quad
y = \begin{bmatrix}200\\ 300\\ 400\end{bmatrix}
$$

### คำนวณพารามิเตอร์:

$$
w = (X^T X)^{-1} X^T y
$$

---

## 🔷 4. ข้อดีของ Normal Equation

- ✅ ไม่ต้องใช้การวนลูป  
- ✅ ได้ผลลัพธ์แม่นยำ  
- ✅ เข้าใจง่าย

---

## 🔷 5. ข้อเสียของ Normal Equation

- ❌ ช้ามากเมื่อฟีเจอร์มีจำนวนมาก  
- ❌ ใช้หน่วยความจำสูง  
- ❌ ใช้ได้เฉพาะกับ Linear Regression

---

## 🔷 6. เปรียบเทียบกับ Gradient Descent

| หัวข้อ              | Normal Equation                | Gradient Descent                      |
|---------------------|--------------------------------|----------------------------------------|
| ความเร็ว            | เร็วสำหรับข้อมูลขนาดเล็ก       | เร็วสำหรับข้อมูลขนาดใหญ่             |
| ความซับซ้อน         | ปานกลาง                        | ต้องเรียนรู้อนุพันธ์และการปรับพารามิเตอร์ |
| ข้อจำกัด            | ใช้ได้เฉพาะ Linear Regression | ใช้ได้กับโมเดลหลากหลาย               |
| Learning Rate       | ไม่ต้องปรับ                    | ต้องปรับ                              |

---
