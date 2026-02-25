import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_data(file_path):
    print("Loading raw data...")
    df = pd.read_csv(file_path)

    print("Cleaning data...")
    # 1. Drop duplicates based on the 'id'
    df = df.drop_duplicates(subset=['id'])

    # 2. Extract Primary Cuisine to prevent dimension explosion in One-Hot Encoding
    df['cuisine'] = df['cuisine'].astype(str)
    df['primary_cuisine'] = df['cuisine'].str.split(',').str[0].str.strip()

    # 3. Clean 'cost' column (e.g., '₹ 200' -> 200)
    df['cost'] = df['cost'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

    # 4. Clean 'rating' column (Replace '--' with NaN)
    df['rating'] = df['rating'].replace('--', np.nan)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Impute missing ratings with the overall median (around 3.9) so we don't lose half the dataset
    df['rating'] = df['rating'].fillna(df['rating'].median())

    # 5. Clean 'rating_count'
    def parse_rating_count(val):
        val = str(val).lower()
        if 'too few' in val or 'nan' in val:
            return 0
        elif 'k+' in val:
            return float(val.replace('k+ ratings', '').strip()) * 1000
        else:
            return float(val.replace('+ ratings', '').strip())
            
    df['rating_count'] = df['rating_count'].apply(parse_rating_count)

    # 6. Drop rows where essential columns are still missing (like city)
    core_cols = ['id', 'name', 'city', 'rating', 'rating_count', 'cost', 'primary_cuisine']
    df = df.dropna(subset=core_cols)
    
    # Optional: Keep only necessary columns to keep the Streamlit app lightweight
    columns_to_keep = ['id', 'name', 'city', 'rating', 'rating_count', 'cost', 'primary_cuisine', 'link', 'address']
    df = df[columns_to_keep]

    # Save the cleaned dataset
    df.reset_index(drop=True, inplace=True)
    df.to_csv('cleaned_data.csv', index=False)
    print(f"Cleaned data saved to 'cleaned_data.csv'. Shape: {df.shape}")

    print("Preprocessing and Encoding...")
    categorical_features = ['city', 'primary_cuisine']
    numerical_features = ['rating', 'rating_count', 'cost']

    # Initialize One-Hot Encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(df[categorical_features])
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])

    # Combine encoded categorical and scaled numerical data
    encoded_df = pd.DataFrame(
        encoded_categorical, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    numerical_df = pd.DataFrame(
        scaled_numerical, 
        columns=numerical_features
    )
    
    final_encoded_data = pd.concat([encoded_df, numerical_df], axis=1)

    # Save the encoded dataset and transformers
    final_encoded_data.to_csv('encoded_data.csv', index=False)
    print("Encoded data saved to 'encoded_data.csv'")

    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Encoder and Scaler saved successfully as Pickle files!")

if __name__ == "__main__":
    prepare_data('swiggy.csv')