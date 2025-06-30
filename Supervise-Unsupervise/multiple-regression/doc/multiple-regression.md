## Multiple regresion
---

# 🔍 Overview: Multiple Linear Regression

- In the earlier version of linear regression, we used only **one feature** (e.g., size of the house) to predict the output (e.g., price of the house). but for **Multiple Linear Regression** we use **multiple feature**  such as:
Size of the house (`x₁`), Number of bedrooms (`x₂`), Number of floors (`x₃`) and Age of the house (`x₄`) for predict output value


### 📚 Notation
# 📊 Multiple Features (Variables)


| Size in feet² (𝑥₁) | Number of bedrooms (𝑥₂) | Number of floors (𝑥₃) | Age of home in years (𝑥₄) | Price ($1000's) |
|:------------------:|:------------------------:|:----------------------:|:--------------------------:|:----------------:|
|       2104         |            5             |           1            |            45              |       460        |
|       1416         |            3             |           2            |            40              |       232        |
|       1534         |            3             |           2            |            30              |       315        |
|        852         |            2             |           1            |            36              |       178        |
|       ...          |           ...            |          ...           |           ...              |       ...        |


- \( x_j \) = jᵗʰ feature  
- \( n = 4 \) = number of features  
- \( \vec{x}^{(i)} \) = features of the iᵗʰ training example  
- \( x_j^{(i)} \) = value of feature \( j \) in the iᵗʰ training example  

### 🟦 Example from the table:

- \( i = 2 \) → second training example  
- \( \vec{x}^{(2)} = [1416,\ 3,\ 2,\ 40] \)  
- \( x_3^{(2)} = 2 \)


---

# 🧠 Model Definition

- **Old model (1 feature)**:
  \[
  f_{w,b}(x) = wx + b
  \]

- **New model (n features)**:
  \[
  f_{w,b}(x) = w₁x₁ + w₂x₂ + \dots + wₙxₙ + b
  \]

- **Compact vector notation using dot product**:
  \[
  f_{w,b}(x) = \vec{w} \cdot \vec{x} + b
  \]


> Dot Product
>\[
>\vec{w} \cdot \vec{x} = \sum_{j=1}^{n} w_j x_j
>\]


# 🏠 Example Interpretation (House Prices)

\[
f(x) = 0.1x₁ + 4x₂ + 10x₃ - 2x₄ + 80
\]

- `0.1`: price increases by $100 per square foot
- `4`: price increases by $4,000 per bedroom
- `10`: price increases by $10,000 per floor
- `-2`: price **decreases** by $2,000 per year of age
- `80`: base price in $1,000s (i.e., $80,000)


# ⚡ Vectorization 
## Parameters and Features

\[
\vec{w} = [w_1, w_2, w_3] \quad \text{with } n = 3
\]

\[
b \text{ is a number}
\]

\[
\vec{x} = [x_1, x_2, x_3]
\]

**Note:**

- Linear algebra counts from **1**.
- Code (NumPy/Python) counts from **0**.

```python
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```

---

## Without Vectorization

\[
f_{\vec{w}, b}(\vec{x}) = \sum_{j=1}^{n} w_j x_j + b
\]



 Without Vectorization (Expanded Formula, \( n = 100{,}000 \))

\[
f_{\vec{w}, b}(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
\]

```python
f = w[0] * x[0] + 
    w[1] * x[1] + 
    w[2] * x[2] + b
```

😢 Manual, repetitive, and error-prone

In Python:

```python
f = 0
for j in range(0, n):  # j = 0 to n-1
    f = f + w[j] * x[j]
f = f + b
```
## Vectorization (Efficient Way)

\[
f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b
\]

```python
f = np.dot(w, x) + b
```

😀 Clean and fast with NumPy


- **Vectorization** is a technique to make code:
  - **Shorter** (easier to read/write)
  - **Faster** (more computationally efficient)
- Uses **linear algebra operations** and libraries like **NumPy**
- Can take advantage of **parallel hardware** like **CPUs** and **GPUs**

---
## Without Vectorization

```python
for j in range(0, 16):
    f = f + w[j] * x[j]
```

At different time steps:

- \( t_0 \): \( f + w[0] \cdot x[0] \)
- \( t_1 \): \( f + w[1] \cdot x[1] \)
- ...
- \( t_{15} \): \( f + w[15] \cdot x[15] \)

⚠️ **Sequential**, not optimized


## Vectorization

```python
np.dot(w, x)
```

### Computation flow:

\[
\begin{align*}
\text{At } t_0: & \quad 
\begin{bmatrix}
w[0] & w[1] & \dots & w[15]
\end{bmatrix}
\times
\begin{bmatrix}
x[0] \\ x[1] \\ \vdots \\ x[15]
\end{bmatrix}
\quad \text{(in parallel)}
\\[10pt]
\text{At } t_1: & \quad 
w[0] \cdot x[0] + w[1] \cdot x[1] + \dots + w[15] \cdot x[15]
\end{align*}
\]

✅ **Efficient → scales to large datasets**
# 🧾 Gradient descent

## Gradient Descent

\[
\vec{w} = (w_1, w_2, \ldots, w_{16}) \quad \text{(parameters)}
\]
\[
\vec{d} = (d_1, d_2, \ldots, d_{16}) \quad \text{(derivatives)}
\]

```python
w = np.array([0.5, 1.3, ..., 3.4])
d = np.array([0.3, 0.2, ..., 0.4])
```

### Compute:

\[
w_j = w_j - 0.1 \cdot d_j \quad \text{for } j = 1 \ldots 16
\]

📝 `0.1` is the **learning rate** \( \alpha \)


### Without Vectorization

\[
\begin{align*}
w_1 &= w_1 - 0.1 d_1 \\
w_2 &= w_2 - 0.1 d_2 \\
&\vdots \\
w_{16} &= w_{16} - 0.1 d_{16}
\end{align*}
\]

```python
for j in range(0, 16):
    w[j] = w[j] - 0.1 * d[j]
```

### With Vectorization

\[
\vec{w} = \vec{w} - 0.1 \cdot \vec{d}
\]

```python
w = w - 0.1 * d
```

➡️ Performs the operation on all elements **in parallel** → ✅ **Efficient for large models**

# 🧾 Terminology

- **Multiple Linear Regression**: Linear regression with **multiple input features**
- **Univariate Regression**: Linear regression with a **single feature**
- Note: **"Multivariate regression"** refers to a different concept (not used here)


## Gradient descent for multiple linear regression
