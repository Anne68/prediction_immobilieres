import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.compose import ColumnTransformer

loaded_pipeline = joblib.load('/home/vicky/Projet_prediction_immobilier_Arturo/knn_regressor_pipeline.joblib')

# Définir la fonction de prédiction
def predict_house_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]],
                             columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

    prediction = loaded_pipeline.predict(input_data)
    return prediction[0]

# Charger le modèle pré-entraîné (à adapter selon votre modèle)
# Exemple : Chargement d'un modèle de RandomForestRegressor
#pipeline = Pipeline(steps=[('regressor', RandomForestRegressor())])  # Modifier avec votre propre pipeline

# Page d'accueil de l'application web
st.title('Prédiction de prix immobiliers')

# Interface utilisateur pour saisir les données
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Âge médian du logement')
total_rooms = st.number_input('Nombre total de pièces')
total_bedrooms = st.number_input('Nombre total de chambres')
population = st.number_input('Population')
households = st.number_input('Ménages')
median_income = st.number_input('Revenu médian')
ocean_proximity = st.selectbox('Proximité de l\'océan', ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

# Bouton pour lancer la prédiction
if st.button('Prédire le prix'):
    prediction_result = predict_house_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)
    st.success(f'Le prix prédit du bien immobilier est : {prediction_result:.2f} $')



import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import r2_score

csv_file = st.file_uploader("Téléchargez un fichier CSV", type=['csv'])

if csv_file is not None:
    df = pd.read_csv(csv_file)

    # Sélectionnez les colonnes que vous souhaitez utiliser pour les prédictions
    X = df


    # Chargez le modèle à partir du fichier joblib
    loaded_model = joblib.load('/home/vicky/Projet_prediction_immobilier_Arturo/modele_regr_Ryan.joblib')

    # Ajoutez une nouvelle colonne avec les prédictions
    df['prediction'] = loaded_model.predict(X)

    # Calculez le R2 score en utilisant la variable dépendante réelle et les prédictions
    y = df['median_house_value']
    r2 = r2_score(y, df['prediction'])
    st.write('R2 score: ', r2)

    st.dataframe(df.head())
    df.to_csv("donnees_immo_prediction.csv", index=False)
