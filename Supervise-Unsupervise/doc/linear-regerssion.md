## Linear regression

---

###  Definition
![alt text](image.png)
ref : https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/
**Linear regression** is a supervised learning algorithm that models the relationship between:
- One or more independent variables (features, denoted as X)
- A dependent variable (target, denoted as y) 
by fitting a linear equation to the observed data.

### Key Characteristics:
1. **Linear Relationship**: Assumes a straight-line relationship between variables
2. **Equation Form**: 
   - Simple: `y = wX + b` (one feature)
   - Multiple: `y = w₁X₁ + w₂X₂ + ... + b` (multiple features)
   where:
   - `w` = coefficients (weights)
   - `b` = y-intercept (bias term)

3. **Objective**: Minimizes the difference between predicted (ŷ) and actual values (y) using:
   - **Ordinary Least Squares (OLS)**: Minimizes sum of squared residuals

4. **Use Cases**:
   - Predicting continuous values (prices, temperatures)
   - Understanding feature importance
   - Trend analysis



### Key Notation

#### Example Housing  Dataset

| Index | Size (ft²) | Price ($1000's) |
|-------|------------|----------------|
| 1     | 2104       | 400            |
| 2     | 1416       | 232            |
| 3     | 1534       | 315            |
| 4     | 852        | 178            |
| ...   | ...        | ...            |
| 47    | 3210       | 870            |

## Variables


| Symbol | Meaning | Example |
|--------|---------|---------|
| `x` | Input feature (independent variable) | House size (2104 sq ft) |
| `y` | Output target (dependent variable) | Price ($400,000) |
| `m` | Number of training examples | 47 houses |
| `(xⁱ, yⁱ)`    | i-th training example | `(2104, 400000)` |
| `ŷ` | Predicted output | `ŷ = f(x)` |


# Supervised Learning & Linear Regression Process  

![alt text](<../img/Screenshot 2568-06-13 at 15.43.50.png>)

## How Supervised Learning Works  
1. **Input**: Training set with:  
   - Features (x): Input variables (e.g., house size)  
   - Targets (y): True output values (e.g., house price)  

2. **Output**: A function **`f`** (the *model*) that:  
   - Takes new input **`x`**  
   - Predicts output **`ŷ`** (estimated value)  

---

## Linear Regression Model  
- **Equation**: `f(x) = w·x + b`  
  - **`w`**: Weight (slope)  
  - **`b`**: Bias (y-intercept)  
- **Goal**: Fit a straight line to minimize prediction errors.  

### Key Terms  
- **Univariate Linear Regression**: Single input variable (e.g., house size).  
- **Non-linear vs. Linear**:  
  - Linear: Simple, foundational (straight line).  
  - Non-linear: Used for complex patterns (curves).  

---

## Next Steps  
1. **Cost Function**: Measures prediction accuracy (e.g., residual sum of squares).  
2. **Lab Exercise**: Optional Python lab to experiment with `w` and `b`.  

> 💡 **Why Linear Regression?**  
> Simple, interpretable, and the basis for advanced models.  


# 📘 Cost Function in Linear Regression – Summary

## 🔧 Model Overview
To train a linear regression model, the first essential step is to define a **cost function** that evaluates how well the model is performing.

The **linear model** is: f_{w,b}(x) = wx + b


where:
- `w` (weight) and `b` (bias) are the **parameters** of the model.
- These are adjusted during training to improve the model's predictions.

## 🧮 Examples of `w` and `b`:
- `w = 0`, `b = 1.5` → constant prediction (horizontal line)
- `w = 0.5`, `b = 0` → increasing line with slope 0.5
- `w = 0.5`, `b = 1` → same slope, shifted up

## 🎯 Objective
Choose `w` and `b` so that the predictions `ŷ^(i)` are **close** to the actual values `y^(i)` in the training set.

## 📊 Measuring Accuracy

- **Prediction error** for each training example: error = ŷ^(i) - y^(i)


- **Squared error**: (ŷ^(i) - y^(i))²


- **Cost function over all `m` examples**: J(w, b) = (1 / 2m) * Σ (f_w,b(x^(i)) - y^(i))²


This is called the **squared error cost function**, and it measures the average squared difference between the predicted and actual values.

## ✅ Key Points
- `ŷ^(i)` is the predicted output, `y^(i)` is the true target.
- The division by `2m` is conventional to simplify gradient descent computations.
- The goal of training is to **minimize** the cost function `J(w, b)`.

---

# Cost Function in Linear Regression  

## **Purpose of Cost Function**  
- Measures how well the model (**`f(x) = w·x + b`**) fits the training data.  
- Guides the selection of optimal parameters **`w`** (weight) and **`b`** (bias).  

---

## **Key Concepts**  
### 1. **Model Parameters**  
- **`w`**: Controls the slope of the line.  
- **`b`**: Controls the y-intercept.  
- *Example*:  
  - If **`w=0.5, b=1`**, the line passes through (0,1) with slope 0.5.  

### 2. **Squared Error Cost Function**  
- **Formula**:  J(w,b) = (1/2m) * Σ (ŷⁱ - yⁱ)²

- **`ŷⁱ`**: Prediction for i-th example (**`f(xⁱ) = w·xⁱ + b`**).  
- **`yⁱ`**: Actual target value.  
- **`m`**: Number of training examples.  

- **Why Squared Error?**  
- Penalizes larger errors more heavily.  
- Averages errors over all examples (avoids scaling with dataset size).  

### 3. **Intuition**  
- **Goal**: Minimize **`J(w,b)`** to find the best-fit line.  
- **Small `J`**: Predictions are close to true values.  
- **Large `J`**: Poor fit (predictions deviate significantly).  

---

## **Next Steps**  
- Visualize how **`J(w,b)`** changes with different **`w`** and **`b`**.  
- Use optimization techniques to minimize the cost function.  

> 💡 **Why Divide by 2m?**  
> Makes gradient calculations cleaner (derivative of `x²` is `2x`, so the `2` cancels out).  
Key Takeaways
The cost function quantifies model accuracy.

Adjusting w and b directly impacts J(w,b).

Minimizing J(w,b) is the core of training linear regression models.




