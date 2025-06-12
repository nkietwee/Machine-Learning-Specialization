# 🧠 Unsupervised Learning: Summary

## 🔍 What is Unsupervised Learning?

- Unlike supervised learning, **unsupervised learning** works with **input data `x` only** — there are **no output labels `y`**.
- The goal is to **discover hidden structure**, **patterns**, or **insights** within the data.
- The algorithm is not told what to look for — it must **figure things out on its own**.

---

## 🧩 Main Idea

> In supervised learning, you're given **`x → y`**.  
> In unsupervised learning, you're only given **`x`**, and must find structure or patterns in the data.

---

## 📊 Example: Tumor Dataset

- Supervised: Tumor size + age → label (benign/malignant)
- **Unsupervised**: Tumor size + age (no labels) → Group similar patients automatically

---

## 📌 Types of Unsupervised Learning

### 1. 🔵 Clustering

- Groups similar examples into **clusters**.
- **Example**: Google News groups related news articles using word similarity.
- No human labels are given; the algorithm **learns what belongs together** on its own.

#### 📰 Clustering in Action

- **Google News**: Articles with similar words like *panda*, *twins*, *zoo* are grouped together.
- No one tells the algorithm to group by "panda" — it **learns it automatically**.

#### 🧬 DNA Clustering

- **DNA microarray**: Rows = genes, Columns = individuals.
- Colors show gene activity.
- Clustering groups people by gene expression patterns (e.g., Type 1, Type 2, Type 3).
- No prior labels — just structure found in data.

#### 🛍️ Market Segmentation

- **Customer data**: Group users by behavior or motivation.
- Used by DeepLearning.AI to find groups like:
  - Learners seeking skills
  - Career-driven learners
  - People tracking AI trends

---

### 2. 🚨 Anomaly Detection (Preview)

- Finds **unusual or rare** events in the data.
- Useful in **fraud detection**, system failures, etc.

---

### 3. 📉 Dimensionality Reduction (Preview)

- **Compresses large datasets** into fewer dimensions.
- Keeps as much important information as possible.
- Helps with visualization and efficiency.

---

## 🧠 Supervised vs. Unsupervised Examples

| Example                        | Type                |
|-------------------------------|---------------------|
| Spam detection (labeled data) | Supervised          |
| News article clustering        | Unsupervised        |
| Market segmentation            | Unsupervised        |
| Diagnosing diabetes (labels)   | Supervised          |

---

## 💬 Summary

- **Unsupervised Learning**: Discover hidden patterns in **unlabeled data**.
- Common tasks include:
  - ✅ Clustering
  - ✅ Anomaly Detection
  - ✅ Dimensionality Reduction
- Algorithms learn **without supervision**, discovering **natural groupings and structures** in data.

---

## 📚 Coming Up Next

- Deep dives into:
  - **Anomaly Detection**
  - **Dimensionality Reduction**
- Plus: Hands-on experience with **Jupyter Notebooks** for machine learning.
