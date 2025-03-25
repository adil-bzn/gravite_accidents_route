#Import des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Chargement des datasets
caracteristiques = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\caracteristiques.csv"
lieux = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\lieux.csv"
usagers = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\usagers.csv"
vehicules = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\vehicules.csv"
description_variables =r"C:\Users\adilb\Downloads\Projets accidents - Description des variables.xlsx"
df_caracteristiques = pd.read_csv(caracteristiques, encoding = "ISO-8859-1", index_col=0)
df_lieux = pd.read_csv(lieux, encoding = "ISO-8859-1",index_col=0)
df_usagers = pd.read_csv(usagers, encoding = "ISO-8859-1",index_col=0)
df_vehicules = pd.read_csv(vehicules, encoding = "ISO-8859-1",index_col=0)
df_description_variables = pd.read_excel(description_variables)

#Titre du streamlit
st.title("Projet accidents - Prédiction de la gravité")

#Configuration bandeau gauche de la page
    #Sommaire et navigation
st.sidebar.title("Sommaire")
pages=["Contexte", "Exploration", "Data Visualisation", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
    #Auteurs
st.sidebar.title("Auteurs :")
st.sidebar.write("Adil Bouziane")
st.sidebar.write("Malika Sadki")
st.sidebar.write("Anouar Ouro-Sao")

if page == pages[1] :

    st.write("Description du BAAC")
    st.write("Description de la variable gravité")
    st.write("Signification de chaque gravité au sens du BAAC")
    st.write("Description de chaque dataset")
    
    checkbox_df_selected = st.selectbox("Afficher les caractéristiques de l'un des 4 datasets :", ("df_caracteristiques","df_lieux","df_usagers","df_vehicules"), index=None, placeholder="Sélectionner un dataset")
    if checkbox_df_selected == "df_caracteristiques" :
        st.dataframe(df_caracteristiques.head(5))

    if checkbox_df_selected == "df_lieux" :
        st.dataframe(df_lieux.head(5))

    if checkbox_df_selected == "df_usagers" :
        st.dataframe(df_usagers.head(5))

    if checkbox_df_selected == "df_vehicules" :
        st.dataframe(df_vehicules.head(5))

    #A FAIRE : Mettre l'image de la jointure entre les tables

    st.write("Explication sur le choix de supprimer les années >= à 2018")
    st.write("Relever la présence de d oublons dans le df_usagers")

    df_usagers=df_usagers.drop_duplicates(keep="first")

    st.write("Mise à disposition du dictionnaire de chaque variable")
    st.dataframe(df_description_variables, hide_index=True)

if page == pages[2] :