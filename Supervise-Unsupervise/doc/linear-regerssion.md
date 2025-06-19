## Linear Regression

---

### Definition

![alt text](image.png)  
**Reference**: [GeeksforGeeks – Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)

**Linear regression** is a supervised learning algorithm that models the relationship between:
- One or more **independent variables** (features, denoted as `X`)
- A **dependent variable** (target, denoted as `y`)  
by fitting a linear equation to the observed data.

---

### How Supervised Learning Works  

1. **Input**: Training set with:  
   - **Features (`x`)**: Input variables (e.g., house size)  
   - **Targets (`y`)**: True output values (e.g., house price)  

2. **Output**: A function **`f`** (the *model*) that:  
   - Takes new input **`x`**  
   - Predicts output **`ŷ`** (estimated value)

---

### Key Notation

#### Example: Housing Dataset

| Index | Size (ft²) | Price ($1000's) |
|-------|------------|-----------------|
| 1     | 2104       | 400             |
| 2     | 1416       | 232             |
| 3     | 1534       | 315             |
| 4     | 852        | 178             |
| ...   | ...        | ...             |
| 47    | 3210       | 870             |

#### Variable Definitions

| Symbol     | Meaning                            | Example                |
|------------|------------------------------------|------------------------|
| `x`        | Input feature (independent variable) | House size (2104 sq ft) |
| `y`        | Output target (dependent variable)   | Price ($400,000)        |
| `m`        | Number of training examples          | 47 houses               |
| `(xⁱ, yⁱ)` | i-th training example                | (2104, 400000)          |
| `ŷ`        | Predicted output                     | `ŷ = f(x)`              |

---

### Key Characteristics of Linear Regression
![alt text](image-1.png)

1. **Linear Relationship**  
   - Assumes a straight-line relationship between variables

2. **Equation Form**  
   - Simple Linear Regression:  
     ```
     y = wX + b
     ```
     where:  
     - `w` = coefficient (weight)  
     - `b` = y-intercept (bias term)

3. **Objective**  
   - Minimize the difference between predicted values `ŷ` and actual values `y` using a **cost function** (e.g., MSE (Mean Squared Error))

4. **Use Cases**  
   - Predicting continuous values (e.g., prices, temperatures)  
   - Understanding feature importance  
   - Trend analysis

---

![Linear Regression Example](../img/Screenshot%202568-06-13%20at%2015.43.50.png)


# 📉 Cost Function in Linear Regression

The **cost function** is a mathematical tool used to quantify the error between predicted outputs and actual target values in a regression model. It serves as a measure of how well the model's parameters fit the training data.

- **Purpose**: To evaluate the performance of the model by measuring the discrepancy between predicted and actual outcomes.
- **Objective**: Minimize the cost function in order to optimize the model parameters and improve prediction accuracy.





### ✅ Steps to Construct the Cost Function:

1. **Error**:
   \[
   \text{Error}^{(i)} = \hat{y}^{(i)} - y^{(i)}
   \]

2. **Squared Error**:
   \[
   \left( \hat{y}^{(i)} - y^{(i)} \right)^2
   \]

3. **Total Squared Error**:
   \[
   \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
   \]

4. **Average Squared Error**:
   \[
   \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
   \]

5. **Final Cost Function (J)**:
   \[
   J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
   \]

> 📌 `J(w, b)` is called the **squared error cost function** and is widely used in linear regression tasks.

> 💡 **Why Divide by 2m?**  
> Dividing by 2m is a convention that makes the math easier and more elegant during model training.

---

## 📘 Linear Regression Summary

**🔹 Model:**  
\[
f_{w,b}(x) = wx + b
\]

**🔹 Parameters:**  
\[
w, \; b
\]

**🔹 Cost Function:**  
\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
\]

**🔹 Objective:**  
\[
\min_{w, b} J(w, b)
\]






