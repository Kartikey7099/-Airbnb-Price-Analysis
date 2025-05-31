import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import pydeck as pdk
import ast

st.set_page_config(page_title="Airbnb Price Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("listings.csv")
    columns_to_keep = [
        'id', 'name', 'host_id', 'host_name', 'neighbourhood',
        'latitude', 'longitude', 'room_type', 'price', 'minimum_nights',
        'number_of_reviews', 'reviews_per_month', 'availability_365', 'amenities'
    ]
    df = df[columns_to_keep]
    df = df[df['price'].notnull() & df['reviews_per_month'].notnull()]
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['host_name'].fillna('Unknown', inplace=True)
    df['neighbourhood'].fillna('Unknown', inplace=True)
    df = df[(df['price'] > 0) & (df['price'] < 1000)]
    df = df[df['minimum_nights'] <= 365]

    def count_amenities(amenities_str):
        try:
            amenities_list = ast.literal_eval(amenities_str)
            return len(amenities_list)
        except:
            return 0
    df['amenities_count'] = df['amenities'].apply(count_amenities)

    le_room = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    neigh_freq = df['neighbourhood'].value_counts().to_dict()
    df['neighbourhood_encoded'] = df['neighbourhood'].map(neigh_freq)

    np.seterr(divide='ignore')
    df['log_price'] = np.log(df['price'])
    with np.errstate(divide='ignore', invalid='ignore'):
        df['review_rate'] = df['number_of_reviews'] / df['availability_365']
        df['review_rate'].replace([np.inf, -np.inf], 0, inplace=True)
        df['review_rate'].fillna(0, inplace=True)

    return df, le_room

@st.cache_data
def train_models(df):
    selected_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'availability_365', 'amenities_count',
        'room_type_encoded', 'neighbourhood_encoded', 'review_rate'
    ]
    X = df[selected_features]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_imputed, y_train)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_imputed, y_train)

    # HistGradientBoosting with GridSearch
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_iter': [100, 200],
        'max_depth': [None, 5, 10]
    }
    gb_base = HistGradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb_base, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)  # no imputation needed for HGB
    gb_best = grid_search.best_estimator_

    return lr, rf, gb_best, X_test_imputed, X_test, y_test, imputer, grid_search.best_params_, grid_search.best_score_

def format_price(price):
    return f"₹{price:,.2f}"

df, le_room = load_data()
lr_model, rf_model, gb_model, X_test_imputed, X_test_raw, y_test, imputer, best_gb_params, best_gb_score = train_models(df)

st.title("Airbnb Price Analysis & Booking Trends")

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Model Performance & Feature Importance", "Predict New Listing Price"])

with tab1:
    st.header("Data Filtering and Visualizations")
    neighs = sorted(df['neighbourhood'].unique())
    room_types = df['room_type'].unique()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_neigh = st.selectbox("Select Neighbourhood", neighs)
    with col2:
        selected_room = st.selectbox("Select Room Type", room_types)
    with col3:
        price_min = int(df['price'].min())
        price_max = int(df['price'].max())
        selected_price_range = st.slider("Select Price Range", price_min, price_max, (50, 300))

    filtered_df = df[
        (df['neighbourhood'] == selected_neigh) &
        (df['room_type'] == selected_room) &
        (df['price'] >= selected_price_range[0]) &
        (df['price'] <= selected_price_range[1])
    ]

    st.subheader(f"Filtered Listings: {filtered_df.shape[0]} found")

    st.markdown("### Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df['price'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Review Rate vs Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=filtered_df, x='review_rate', y='price', ax=ax)
    st.pyplot(fig)

    st.markdown("### Amenities Count vs Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=filtered_df, x='amenities_count', y='price', alpha=0.7, ax=ax)
    st.pyplot(fig)

    st.markdown("### Listing Locations (Hover over dots for price)")
    tooltip = {
        "html": "<b>Name:</b> {name} <br> <b>Price:</b> ₹{price}",
        "style": {"color": "white"},
    }
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position='[longitude, latitude]',
        get_radius=100,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['latitude'].mean(),
        longitude=filtered_df['longitude'].mean(),
        zoom=11,
        pitch=50,
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r)

    st.markdown("### Correlation Heatmap")
    corr = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
               'availability_365', 'amenities_count', 'review_rate']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Model Performance & Feature Importance")
    st.markdown("""
    **Metrics Explanation:**  
    - RMSE (Root Mean Squared Error): Lower values mean better fit; measures average error magnitude.  
    - R² Score: How well model explains variance in the data (1 = perfect, 0 = no better than mean).
    """)

    y_pred_lr = lr_model.predict(X_test_imputed)
    y_pred_rf = rf_model.predict(X_test_imputed)
    y_pred_gb = gb_model.predict(X_test_raw)  # no imputation needed

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)

    st.subheader("Linear Regression")
    st.write(f"RMSE: {rmse_lr:.3f}")
    st.write(f"R² Score: {r2_lr:.3f}")

    st.subheader("Random Forest Regressor")
    st.write(f"RMSE: {rmse_rf:.3f}")
    st.write(f"R² Score: {r2_rf:.3f}")

    st.subheader("Tuned HistGradientBoosting Regressor")
    st.write(f"RMSE: {rmse_gb:.3f}")
    st.write(f"R² Score: {r2_gb:.3f}")
    st.write("Best Parameters:")
    st.json(best_gb_params)
    st.write(f"Best Cross-Validation R²: {best_gb_score:.3f}")

    st.markdown("### Feature Importance (Random Forest)")
    importances_rf = rf_model.feature_importances_
    features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'availability_365', 'amenities_count',
        'room_type_encoded', 'neighbourhood_encoded', 'review_rate'
    ]
    imp_rf_df = pd.DataFrame({'Feature': features, 'Importance': importances_rf}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=imp_rf_df, ax=ax)
    st.pyplot(fig)

    st.markdown("### Feature Importance (HistGradientBoosting Regressor - Permutation Importance)")
    perm_result = permutation_importance(gb_model, X_test_raw, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances_gb = perm_result.importances_mean
    imp_gb_df = pd.DataFrame({'Feature': features, 'Importance': importances_gb}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=imp_gb_df, ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("Predict Price for New Listing")

    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=df['latitude'].mean())
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=df['longitude'].mean())
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
    reviews_per_month = st.number_input("Reviews Per Month", min_value=0.0, max_value=50.0, value=1.0)
    availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=180)
    amenities_count = st.number_input("Number of Amenities", min_value=0, max_value=50, value=5)
    room_types = df['room_type'].unique()
    neighs = sorted(df['neighbourhood'].unique())
    room_type_input = st.selectbox("Room Type", room_types)
    neighbourhood_input = st.selectbox("Neighbourhood", neighs)

    room_type_encoded_input = le_room.transform([room_type_input])[0]
    neighbourhood_encoded_input = df['neighbourhood'].value_counts().to_dict().get(neighbourhood_input, 0)
    review_rate_input = number_of_reviews / availability_365 if availability_365 > 0 else 0

    input_features = np.array([
        latitude, longitude, minimum_nights, number_of_reviews, reviews_per_month,
        availability_365, amenities_count, room_type_encoded_input, neighbourhood_encoded_input,
        review_rate_input
    ]).reshape(1, -1)

    model_choice = st.radio("Select Model for Prediction", ("Linear Regression", "Random Forest", "HistGradientBoosting"))

    if st.button("Predict Price"):
        if model_choice == "Linear Regression":
            input_imputed = imputer.transform(input_features)
            pred_log = lr_model.predict(input_imputed)[0]
        elif model_choice == "Random Forest":
            input_imputed = imputer.transform(input_features)
            pred_log = rf_model.predict(input_imputed)[0]
        else:  # HistGradientBoosting
            pred_log = gb_model.predict(input_features)[0]

        pred_price = np.exp(pred_log)
        st.success(f"Predicted Price: {format_price(pred_price)}")

        # Option to export input and prediction as CSV
        export_df = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'minimum_nights': [minimum_nights],
            'number_of_reviews': [number_of_reviews],
            'reviews_per_month': [reviews_per_month],
            'availability_365': [availability_365],
            'amenities_count': [amenities_count],
            'room_type': [room_type_input],
            'neighbourhood': [neighbourhood_input],
            'review_rate': [review_rate_input],
            'predicted_price': [pred_price]
        })
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Prediction as CSV", csv, "prediction.csv", "text/csv")

