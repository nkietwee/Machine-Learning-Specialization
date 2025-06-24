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

The **cost function** is a mathematical tool used to **quantify the error between predicted outputs and actual target values** in a regression model. It serves as a measure of how well the model's parameters fit the training data.

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

# Gradient Descent Overview

## Introduction
- Gradient descent is a systematic algorithm to **find optimal parameters (w, b) that minimize the cost function J(w, b).**
- **It's widely used in machine learning, including linear regression and advanced neural networks (deep learning).**

## Key Concepts
1. **General Applicability**:
   - Works for any function, not just linear regression cost functions.
   - Can handle multiple parameters (e.g., w₁, w₂, ..., wₙ, b).

2. **Initialization**:
   - Start with initial guesses (commonly 0 for w and b in linear regression).
   - Iteratively adjust parameters to reduce J(w, b).

3. **Intuition**:
   - Visualize the cost function as a hilly terrain.
   - Goal: Efficiently descend to the lowest point (minimum) by taking steps in the **direction of steepest descent**.

4. **Process**:
   - At each point, compute the best direction to take a small step downhill.
   - Repeat until reaching a (local) minimum.

5. **Local Minima**:
   - Depending on the starting point, gradient descent may converge to different local minima.
   - Example: Two valleys (local minima) with different outcomes based on initialization.

## Next Steps
- The next video will cover the **mathematical implementation** of gradient descent.


# Gradient Descent Implementation

## Core Algorithm
- **Parameter Update Rules**:
\[
  w = w - \alpha \frac{\partial}{\partial w} J(w,b)
\]

\[
  b = b - \alpha \frac{\partial}{\partial b} J(w,b)
\]


### **Key Components**
#### **Learning Rate (α)**
- Controls step size during optimization.
- Too small (e.g., 0.001): Slow convergence.
- Too large (e.g., 1.0): Risk of overshooting minima.

#### Partial Derivatives

- **∂J/∂w**: Gradient with respect to **w**  
- **∂J/∂b**: Gradient with respect to **b**

- Interpretation
	- Points in the direction of **steepest ascent**
	- **Negative gradient** → direction of **descent**

### Simultaneous Updates (✅ Correct Way)

**Update w and b simultaneously by**:
```python
temp_w = w - alpha * derivative_w
temp_b = b - alpha * derivative_b

w = temp_w
b = temp_b
```
- First calculate the updates using old values
- Then apply both updates at the same time


### 🚫 **Non-Simultaneous Updates (❌ Incorrect Way)**
In the **incorrect** method:

```python
temp_w = w - alpha * derivative_w
w = temp_w  # ← w updated too early

temp_b = b - alpha * derivative_b  # ← uses already-updated w
b = temp_b

```

**Problem**: The second update uses already-updated w, making it inconsistent. This leads to a different algorithm with different (and usually worse) behavior.

**Visualization**
```
   Start
     ↓
   Compute Gradients
     ↓
Update Parameters → Check Convergence → Stop
     ↑                             ↓
     └─────── Repeat ←────────────┘
```


![alt text](image-2.png)

# 🔍 Gradient Descent – Deeper Intuition

This lesson dives deeper into **how gradient descent works** and why it makes intuitive sense.

---

## 🧮 Gradient Descent Recap

- **Update rule**:
  
\[
  w = w - \alpha \frac{\partial}{\partial w} J(w,b)
\]


- **α (alpha)**: the **learning rate**, controls the size of the step taken.
- **∂J/∂w**: the **derivative (or partial derivative)** of the cost function, indicating the **slope** at a given point.

---

## 📉 What the Derivative Means

- A **derivative** is the slope of the **tangent line** at a given point on the cost curve.
- The slope helps determine the **direction** and **size** of the parameter update.

---

## 🧭 Intuition from Two Scenarios

1. **Starting on the right side of the curve** (slope > 0):
 - Derivative is **positive**
 - `w := w - α * (positive)` → w **decreases**
 - You move **left**, toward the **minimum** of the cost function

2. **Starting on the left side of the curve** (slope < 0):
 - Derivative is **negative**
 - `w := w - α * (negative)` → w **increases**
 - You move **right**, toward the **minimum**

✅ In both cases, **gradient descent moves you toward the minimum**, reducing the cost.

---

## 🔧 Why This Works

- The **derivative** tells you the direction to move
- The **learning rate** controls how far to move
- Together, the update rule makes sure you take steps **that reduce the cost function**
---
other version
# Gradient Descent Intuition

## Core Concept
Gradient descent iteratively adjusts parameters (w) to minimize cost function J(w) using:
`w := w - α * (d/dw J(w))`

Where:
- `α` = learning rate (step size)
- `d/dw J(w)` = derivative (slope of tangent line at current w)

## Key Insights

### 1. Derivative's Role
- **Positive slope** → Decreases w (moves left)
- **Negative slope** → Increases w (moves right)
- Always pushes w toward the minimum of J(w)

### 2. Visualization
- With 1 parameter (w), J(w) is a 2D curve
- Tangent lines show:
  - Steepness (magnitude of derivative)
  - Direction (sign of derivative)

### 3. Behavior Examples
- **Right-side initialization**:
  - Positive derivative → w decreases → moves toward minimum
- **Left-side initialization**:
  - Negative derivative → w increases → moves toward minimum

### 4. Learning Rate (α)
- Controls step size:
  - Too small → Slow convergence
  - Too large → Risk of overshooting
- Will be explored deeper in next video

## Why It Works
The derivative automatically guides updates in the direction that reduces cost, while α determines how aggressively we follow that direction.

## Next Topic
Detailed examination of learning rate α selection and its impact.


## Learning rate

# 🚀 Understanding the Learning Rate (α) in Gradient Descent

---

## 📌 Importance of α

- The **learning rate (α)** controls the step size in parameter updates.
- Choosing **α poorly** can:
  - Make training **very slow** (if too small)
  - Cause the algorithm to **diverge or oscillate** (if too large)

---

## 🔎 Case 1: α Too Small

- **Example**: α = 0.0000001
- **Effect**:
  - Takes **tiny baby steps**
  - Moves **slowly toward the minimum**
  - **Works**, but takes **many iterations**

---

## ⚠️ Case 2: α Too Large

- Takes **huge steps**, potentially **overshooting** the minimum.
- Can cause the cost to **increase** instead of decrease.
- May result in:
  - **Oscillating** updates
  - **Failure to converge**
  - Even **divergence**

---

## ✅ Case 3: At the Minimum

- At a **local minimum**, the derivative is **zero**.
- Update rule becomes:

w := w - α * 0 → w := w


- No change in parameter → stays at the minimum (which is correct behavior)

---

## 🧠 Self-Correcting Behavior

- As gradient descent approaches a **local minimum**, the **derivative shrinks**.
- Smaller derivative → **smaller update step**, even with a fixed α.
- This causes gradient descent to **naturally slow down** near the minimum and stabilize.

---

## 🔁 Final Recap

- Gradient descent:
- Works **with a fixed α**
- Automatically **adjusts step size** through the derivative
- Can minimize **any cost function**, not just mean squared error

---

# Gradient Descent Learning Rate Analysis

## Core Update Rule
`w := w - α * (d/dw J(w))`

## Learning Rate (α) Effects

### 1. α Too Small (e.g., 0.0000001)
- **Behavior**: Extremely small steps
- **Result**:
  - Slow convergence
  - Many iterations needed
- **Visualization**:
Start ● → · → · → · → · → Minimum
(Tiny baby steps)


### 2. α Too Large
- **Behavior**: Overshooting steps
- **Result**:
- Cost may increase
- Potential divergence
- **Visualization**:
Start ● → ↗ → ↘ → ↗ → ↘ (Oscillations)


### 3. Optimal α
- **Behavior**: Balanced steps
- **Result**:
- Steady convergence
- Reaches minimum efficiently
- **Visualization**:
Start ● → → → → Minimum


## Special Case: At Local Minimum
- **Derivative**: d/dw J(w) = 0
- **Update**: w remains unchanged
- **Implication**: Algorithm naturally stops at minima

## Adaptive Step Size Property
- **Near Minimum**:
- Derivatives decrease → smaller steps
- Automatic "slowdown" effect
- **Benefit**: Fixed α can still work effectively

## Practical Guidelines
| Scenario | Solution | Typical Values |
|----------|----------|----------------|
| Slow convergence | Increase α | 0.01 → 0.1 |
| Divergence | Decrease α | 0.1 → 0.001 |
| Good convergence | Maintain α | 0.01-0.1 |

## Key Insights
1. α controls step size and convergence speed
2. Gradient descent automatically adjusts effective step size near minima
3. Finding good α requires experimentation

> **Next**: Applying gradient descent to linear regression cost function











## 📘 Linear Regression Summary

**🔹 Model:**  
f_{w,b}(x) = wx + b
\[
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






