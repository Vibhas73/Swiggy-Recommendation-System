import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data and Models ---
@st.cache_data
def load_data():
    cleaned_data = pd.read_csv('cleaned_data.csv')
    encoded_data = pd.read_csv('encoded_data.csv')
    return cleaned_data, encoded_data

@st.cache_resource
def load_models():
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return encoder, scaler

# Load everything
try:
    cleaned_df, encoded_df = load_data()
    encoder, scaler = load_models()
except FileNotFoundError:
    st.error("Data or model files not found. Please run data_preparation.py first.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Swiggy Recommender", layout="wide")
st.title("🍔 Swiggy Restaurant Recommendation System")
st.write("Find the perfect place to eat based on your specific tastes and budget!")

# --- Sidebar User Inputs ---
st.sidebar.header("Your Preferences")

# Get unique values for dropdowns (dropping any NaNs and ensuring they are strings to avoid sorting errors)
cities = sorted(cleaned_df['city'].dropna().astype(str).unique().tolist())
cuisines = sorted(cleaned_df['primary_cuisine'].dropna().astype(str).unique().tolist())

# Note: We ensure these default values exist in the list to avoid Streamlit errors
default_city = "Bangalore" if "Bangalore" in cities else cities[0]
default_cuisine = "Indian" if "Indian" in cuisines else cuisines[0]

selected_city = st.sidebar.selectbox("Select City", cities, index=cities.index(default_city))
selected_cuisine = st.sidebar.selectbox("Select Cuisine", cuisines, index=cuisines.index(default_cuisine))
selected_rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
selected_rating_count = st.sidebar.slider("Minimum Rating Count", 0, 10000, 100, 50)
selected_cost = st.sidebar.slider("Maximum Cost for Two (₹)", 50, 2000, 500, 50)

# --- Recommendation Logic ---
if st.sidebar.button("Find Restaurants"):
    with st.spinner('Finding the best matches...'):
        # 1. Create a DataFrame for the user's input
        user_input = pd.DataFrame({
            'city': [selected_city],
            'primary_cuisine': [selected_cuisine],
            'rating': [selected_rating],
            'rating_count': [selected_rating_count],
            'cost': [selected_cost]
        })

        # 2. Preprocess the user input using the saved encoder and scaler
        user_cat_encoded = encoder.transform(user_input[['city', 'primary_cuisine']])
        user_num_scaled = scaler.transform(user_input[['rating', 'rating_count', 'cost']])
        
        # 3. Combine into a single vector
        user_vector = np.hstack((user_cat_encoded, user_num_scaled))

        # 4. Filter the encoded dataset for faster computation (optional but recommended for large datasets)
        # To make it realistic, we filter the dataset to only include restaurants in the selected city first.
        # Otherwise, the top recommendations might be in a different state!
        city_mask = cleaned_df['city'] == selected_city
        city_indices = cleaned_df.index[city_mask].tolist()
        
        if not city_indices:
            st.warning(f"No restaurants found in {selected_city}.")
        else:
            # Subset the encoded data to only the selected city
            encoded_city_df = encoded_df.iloc[city_indices]
            
            # Calculate Cosine Similarity
            similarities = cosine_similarity(user_vector, encoded_city_df.values)
            
            # Get top 10 most similar restaurant indices (relative to the subset)
            top_relative_indices = similarities.flatten().argsort()[::-1][:10]
            
            # Map back to absolute indices in the main cleaned_df
            top_absolute_indices = [city_indices[i] for i in top_relative_indices]

            # 5. Fetch the recommendations
            recommendations = cleaned_df.iloc[top_absolute_indices].copy()
            recommendations['Match Score (%)'] = (similarities.flatten()[top_relative_indices] * 100).round(2)

            # --- Display Results ---
            st.subheader(f"Top 10 Recommendations for {selected_cuisine} in {selected_city}")
            
            # Formatting the dataframe for a cleaner display
            display_cols = ['name', 'primary_cuisine', 'rating', 'cost', 'Match Score (%)']
            st.dataframe(recommendations[display_cols].reset_index(drop=True), use_container_width=True)

            # Detailed Expanders
            st.markdown("### Restaurant Details")
            for index, row in recommendations.iterrows():
                with st.expander(f"🍽️ {row['name']} - ⭐ {row['rating']} (Score: {row['Match Score (%)']}%)"):
                    st.write(f"**Cuisine:** {row['primary_cuisine']}")
                    st.write(f"**Cost for Two:** ₹{row['cost']}")
                    st.write(f"**Total Ratings:** {int(row['rating_count'])}")
                    st.write(f"**Address:** {row['address']}")
                    if pd.notna(row['link']):
                        st.markdown(f"🔗 [Order on Swiggy]({row['link']})")