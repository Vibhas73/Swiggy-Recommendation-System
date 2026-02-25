# 🍔 Swiggy Restaurant Recommendation System

## 📌 Project Overview
The objective of this project is to build a robust recommendation system based on Swiggy restaurant data. The system recommends restaurants to users based on input features such as city, minimum rating, budget (cost for two), and cuisine preferences. The application utilizes Cosine Similarity to generate highly accurate, personalized recommendations and displays the results in an interactive, easy-to-use Streamlit interface.

**Domain:** Recommendation Systems & Data Analytics

## 🛠️ Skills & Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (One-Hot Encoding, StandardScaler, Cosine Similarity)
* **Web Application:** Streamlit
* **Version Control:** Git & GitHub

## 📊 Dataset Description
The dataset consists of restaurant details provided in a CSV format (`swiggy.csv`).
Key features include:
* **Categorical:** `name`, `city`, `cuisine`
* **Numerical:** `rating`, `rating_count`, `cost`
* **Other:** `id`, `link`, `address`

## ⚙️ Project Architecture & Methodology

### 1. Data Understanding and Cleaning
* **Duplicate Removal:** Identified and dropped duplicate rows based on unique restaurant IDs.
* **Missing Values:** Imputed missing ratings (represented as `--`) with the dataset's median to retain over 50% of the data. Dropped rows missing core mandatory features.
* **Text Formatting:** Cleaned the `cost` column by stripping currency symbols (e.g., "₹ 200" to `200`) and parsed `rating_count` text into integers.
* **Output:** Cleaned data saved as `cleaned_data.csv`.

### 2. Data Preprocessing
* **Feature Engineering:** Extracted `primary_cuisine` to prevent high-dimensionality explosion.
* **Encoding:** Applied `OneHotEncoder` to categorical features (`city`, `primary_cuisine`).
* **Scaling:** Applied `StandardScaler` to numerical features (`rating`, `rating_count`, `cost`).
* **Output:** Fully numerical dataset saved as `encoded_data.csv` (indices perfectly matched to the cleaned dataset). Saved transformers as `encoder.pkl` and `scaler.pkl` for real-time app use.

### 3. Recommendation Engine
* **Algorithm:** Implemented **Cosine Similarity** to identify similar restaurants based on user input features.
* **Why Cosine Similarity?** It calculates a precise continuous distance metric (angle) from the user's exact multi-dimensional input, allowing for strict ranking and a percentage-based "Match Score," outperforming hard-boundary clustering like K-Means for this specific personalized use case.
* **Mapping:** Mathematical outputs (indices) from the encoded dataset are mapped back to the non-encoded `cleaned_data.csv` to present readable results to the user.

## 💻 Streamlit Application
The interactive dashboard allows users to dynamically discover restaurants.
* **User Input:** Select `City` and `Cuisine` via dropdowns; filter by `Minimum Rating` and `Maximum Cost` via sliders.
* **Processing:** The app scales and encodes the live user input using the saved `.pkl` models.
* **Output:** Displays the top 10 recommended restaurants in a clean dataframe, alongside expanders containing rich details like the exact address and a direct link to order on Swiggy.

## 🚀 How to Run the Project Locally

**1. Clone the repository:**
```bash
git clone <your-github-repo-url>
cd swiggy_recommendation_project
python data_preparation.py
python -m streamlit run app.py
