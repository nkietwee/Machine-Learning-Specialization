# Alternative to Gradient Descent: Normal Equation

## 🔷 1. What is the Normal Equation?

The **Normal Equation** is a method for finding the parameters of a **Linear Regression** model **without** using iteration (as in Gradient Descent).

### Linear Regression Equation:

$$
\hat{y} = Xw
$$

- \( X \): Matrix of input features (including bias term)  
- \( w \): Vector of parameters (including bias term)  
- \( \hat{y} \): Predicted value from the model

**Goal:** Find \( w \) that makes \( \hat{y} \) as close to \( y \) as possible (i.e., minimize the cost function)

---

## 🔷 2. Calculation Method

### Cost Function:

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

To minimize, set:

$$
\frac{\partial J(w)}{\partial w} = 0
$$

The solution is:

$$
w = (X^T X)^{-1} X^T y
$$

> 📌 This is the **Normal Equation**

---

## 🔷 3. Example

### Housing Price Data:

| House Size (sq. ft.) | Price (thousand THB) |
|----------------------|----------------------|
| 1000                 | 200                  |
| 1500                 | 300                  |
| 2000                 | 400                  |

### Prepare the Data:

$$
X = \begin{bmatrix}1 & 1000\\ 1 & 1500\\ 1 & 2000\end{bmatrix}, \quad
y = \begin{bmatrix}200\\ 300\\ 400\end{bmatrix}
$$

### Compute Parameters:

$$
w = (X^T X)^{-1} X^T y
$$

---

## 🔷 4. Advantages of the Normal Equation

- ✅ No need for looping  
- ✅ Accurate results  
- ✅ Easy to understand

---

## 🔷 5. Disadvantages of the Normal Equation

- ❌ Very slow for large numbers of features  
- ❌ High memory usage  
- ❌ Only works with Linear Regression

---

## 🔷 6. Comparison with Gradient Descent

| Aspect             | Normal Equation                   | Gradient Descent                           |
|--------------------|------------------------------------|---------------------------------------------|
| Speed              | Fast for small datasets            | Fast for large datasets                     |
| Complexity         | Moderate                           | Requires understanding derivatives & tuning |
| Limitations        | Only for Linear Regression         | Works with various models                   |
| Learning Rate      | No need to adjust                  | Needs tuning                                |

---
