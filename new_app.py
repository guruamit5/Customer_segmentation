import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("D:/Customer-Segmentation_End-to-end-Project-master/sales_data_sample.csv")

# Preprocess data
def preprocess_data(CustomerDataset):
    CustomerDataset['ORDERDATE'] = pd.to_datetime(CustomerDataset['ORDERDATE'])
    CustomerDataset['ORDERDATE_NUMERIC'] = (CustomerDataset['ORDERDATE'] - CustomerDataset['ORDERDATE'].min()).dt.days

    to_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY',
                'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME',
                'ORDERNUMBER', "ORDERDATE", "QTR_ID", "ORDERLINENUMBER", "YEAR_ID", "PRODUCTCODE"]
    Dataset = CustomerDataset.drop(to_drop, axis=1)

    numeric_features = ['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP', 'ORDERDATE_NUMERIC']
    categorical_features = ['STATUS', 'PRODUCTLINE', 'DEALSIZE']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ])

    dataset_prepared = preprocessor.fit_transform(Dataset)
    return Dataset, dataset_prepared

# Define silhouette scorer for grid search
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

# Perform KMeans clustering with grid search
def perform_clustering(dataset_prepared):
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'max_iter': [300, 500],
    }

    grid_search = GridSearchCV(KMeans(), param_grid, cv=5, scoring=silhouette_scorer)
    grid_search.fit(dataset_prepared)
    return grid_search.best_params_

# Build and train autoencoder
def train_autoencoder(dataset_prepared):
    input_layer = Input(shape=(dataset_prepared.shape[1],))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(dataset_prepared.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(dataset_prepared, dataset_prepared, epochs=50, batch_size=32)

    encoder = Model(input_layer, encoded)
    return encoder.predict(dataset_prepared)

# Main Streamlit app
def main():
    st.title("Customer Segmentation using Autoencoder and KMeans")

    # Load data
    CustomerDataset = load_data()
    
    # Preprocess data
    Dataset, dataset_prepared = preprocess_data(CustomerDataset)

    # Perform clustering
    best_params = perform_clustering(dataset_prepared)
    st.write("Best parameters for KMeans:", best_params)

    # Train autoencoder and get encoded data
    encoded_data = train_autoencoder(dataset_prepared)

    # Fit KMeans on encoded data
    kmeans_deep = KMeans(n_clusters=best_params['n_clusters'], 
                         init=best_params['init'], 
                         max_iter=best_params['max_iter'])
    kmeans_deep.fit(encoded_data)

    # Assign clusters to the original dataset
    Dataset['Cluster'] = kmeans_deep.labels_

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=encoded_data[:, 0], y=encoded_data[:, 1], hue=Dataset['Cluster'], palette='viridis')
    plt.title('Clusters from Autoencoder Encoded Data')
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
