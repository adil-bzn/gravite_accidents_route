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
description_variables = r"C:\Users\adilb\Downloads\Projets accidents - Description des variables.xlsx"
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
    st.write("\n\n\n\n\n")

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

    st.write("Explication sur le choix de supprimer les années >= à 2018")
    st.write("Relever la présence de doublon dans le df_usagers")
    
    df_usagers=df_usagers.drop_duplicates(keep="first")


    st.image(
            "https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Tables%20et%20jointures.png",width=400)
    st.write("Expliquer la méthodologie de jointure")

    st.write("Mise à disposition du dictionnaire de chaque variable")
    st.dataframe(df_description_variables, hide_index=True)

if page == pages[2] :
    st.write("\n\n\n\n\n\n\n\n\n\n")

    #Titre 1
    st.write("Cartographie des accidents en France Metropolitaine")
    st.write("\n\n")
    st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/Cartographie%20du%20nombre%20d'accident%20en%20France%20M%C3%A9tropolitaine.png")
    #A FAIRE : Bullet point
    #A FAIRE : Explication détaillée

    st.write("\n\n\n\n\n")
    #Titre 1
    st.write("Cartographie des accidents en France Metropolitaine en fonction de la gravité",caption="Répartition du nombre d'accident en France Métropolitaine entre 2005 et 2017.")
    st.write("\n\n")

    checkbox_CarteGravite_selected = st.selectbox("Afficher la répartition d'une des 4 gravités sur la France Métropolitaine :", ("1 - Indemne","2 - Tué","3 - Blessé grave","4 - Blessé léger"), index=None, placeholder="Sélectionner une gravité")
    if  checkbox_CarteGravite_selected == "1 - Indemne" :
        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20indemne%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

    if checkbox_CarteGravite_selected == "2 - Tué" :
        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20tu%C3%A9s%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

    if checkbox_CarteGravite_selected == "3 - Blessé grave" :
        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20bless%C3%A9s%20graves%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

    if checkbox_CarteGravite_selected == "4 - Blessé léger" :
        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20bless%C3%A9s%20l%C3%A9gers%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")
    #A FAIRE : Bullet point
    #A FAIRE : Explication détaillée

    st.write("\n\n\n\n\n")
    #Titre 1
    st.write("Accidents par département")
    st.write("\n\n")
    st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/20%20d%C3%A9partements%20avec%20le%20plus%20d'accident.png", width = 600, caption="Les 20 départements avec le plus d'accident entre 2005 et 2017.")
    st.write("\n")
    st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Image%20Visualisation/20%20d%C3%A9partements%20avec%20le%20moins%20d'accident.png", width = 600, caption="Les 20 départements avec le moins d'accident entre 2005 et 2017.")
    #A FAIRE : Bullet point
    #A FAIRE : Explication détaillée

