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
st.set_page_config(layout="wide")
st.title("Prédiction de la gravité des accidents :collision:")

#Configuration bandeau gauche de la page
    #Sommaire et navigation
st.sidebar.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/DataScientest.png")
st.sidebar.title("Sommaire")
pages=["Contexte", "Exploration", "Data Visualisation", "Pre-processing", "Modélisation", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
    #Auteurs
st.sidebar.title("Auteurs :")
st.sidebar.write("Adil Bouziane")
st.sidebar.write("Malika Sadki")
st.sidebar.write("Anouar Ouro-Sao")

if page == pages[1] :
    st.write("\n\n\n\n\n")

    st.write("### :material/edit_document: Description du BAAC")

    st.write("\n\n")

    with st.expander("**1.** Les forces de l'ordre renseignent les caractéristiques de chaque accident routier", icon=":material/local_police:") :
        st.write('''
            Tout usager impliqué dans un accident corporel de la circulation routière survenu sur le réseau routier ouvert à la circulation publique et impliquant au moins un véhicule doit en avertir les forces de l’ordre (gendarmerie nationale, sécurité publique, Préfecture de police de Paris, Compagnie Républicaine de Sécurité - article R 231-1 du code de la route). \n
            Ces dernières doivent remplir pour chaque accident corporel un Bulletin d’Analyse des Accidents Corporels (BAAC).
        ''')

    st.write("\n\n\n\n\n")
    st.write("---")

    st.write("Description de la variable gravité")

    st.write("\n\n")
    st.write("\n\n\n\n\n")
    st.write("---")

    st.write("Signification de chaque gravité au sens du BAAC")

    st.write("\n\n")
    st.write("\n\n\n\n\n")
    st.write("---")

    st.write("Description de chaque dataset")

    st.write("\n\n")
    st.write("\n\n\n\n\n")
    st.write("---")
    
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
            "https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Tables%20et%20jointures.png",width=400)
    st.write("Expliquer la méthodologie de jointure")

    st.write("Mise à disposition du dictionnaire de chaque variable")
    st.dataframe(df_description_variables, hide_index=True)

if page == pages[2] :

    #Créer 6 sections dans la page
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([":earth_americas: Localisation ", ":motorway: Route", ":partly_sunny: Condition extérieure", ":family: Humain", ":car: Véhicule", ":clock9: Temporalité"])

    #Section "Localisation"

    with tab1 :
        st.write("\n\n\n\n\n\n\n\n\n\n")

        #Titre 1
        st.write("### :material/pin_drop: :material/car_crash: Cartographie des accidents en France Metropolitaine")

        st.write("\n\n")

        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Cartographie%20du%20nombre%20d'accident%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident en France Métropolitaine entre 2005 et 2017.")

        st.write("\n")

        with st.expander("**1.** Les grandes agglomérations apparaissent clairement", icon=":material/location_city:") :
            st.write('''
                Nous remarquons qu’il y a une concentration assez nette des accidents en agglomération. Nous supposons que la concentration de population est un facteur clé dans ce phénomène.
            ''')
        with st.expander("**2.** Le réseau routier reliant les agglomérations est lui aussi visible", icon=":material/road:") :
            st.write('''
                Il est intéressant de noter que le réseau routier reliant les différentes agglomérations est lui aussi assez nette.
            ''')
        
        st.write("\n")
        st.write("A première vue, nous constatons que la majorité des accidents sont concentrées dans les agglomérations et les axes routiers les reliant.")

        st.write("\n\n\n\n\n")
        st.write("---")

        #Titre 1
        st.write("### :material/pin_drop: :material/car_crash: :material/personal_injury: Cartographie des accidents en France Metropolitaine en fonction de la gravité")

        st.write("\n\n")

        st.write("Nous allons explorer la répartition des accidents en fonction de la gravité.")
        #Créer un menu déroulant permettant d'afficher la répartition des accidents en fonction de la gravité sélectionnée par l'utilisateur
        checkbox_CarteGravite_selected = st.selectbox("",("1 - Indemne","2 - Tué","3 - Blessé grave","4 - Blessé léger"), index=None, placeholder="Sélectionner une gravité")
        if  checkbox_CarteGravite_selected == "1 - Indemne" :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20indemne%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

        if checkbox_CarteGravite_selected == "2 - Tué" :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20tu%C3%A9s%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

        if checkbox_CarteGravite_selected == "3 - Blessé grave" :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20bless%C3%A9s%20graves%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")

        if checkbox_CarteGravite_selected == "4 - Blessé léger" :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Cartographie%20nombre%20accident%20et%20nombre%20accident%20par%20gravit%C3%A9%20bless%C3%A9s%20l%C3%A9gers%20en%20France%20M%C3%A9tropolitaine.png",caption="Répartition du nombre d'accident (violet) et du nombre d'accident par gravité en France Métropolitaine entre 2005 et 2017.")
        #A FAIRE : Bullet point
        #A FAIRE : Explication détaillée

        st.write("\n\n\n\n\n")
        st.write("---")

        #Titre 1
        st.write("### :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par département")

        st.write("\n\n")

        col1, col2 = st.columns(2, gap="medium")

        with col1 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/20%20d%C3%A9partements%20avec%20le%20plus%20d'accident.png", width = 600, caption="Les 20 départements avec le plus d'accident entre 2005 et 2017.")
        with col2 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/20%20d%C3%A9partements%20avec%20le%20moins%20d'accident.png", width = 600, caption="Les 20 départements avec le moins d'accident entre 2005 et 2017.")
        
        st.write("\n")

        with st.expander("**1.** Problème de data quality - Présence d'un 0 à la fin de chaque code département", icon=":material/data_alert:") :
            st.write('''
                Les données de département ne sont pas correctes dans notre dataset. Un « 0 » est présent à la fin du code de département.\n
                Il faudra prendre en compte cet élément dans notre preprocessing.
            ''')
        with st.expander("**2.** Il existe une corrélation entre nombre d'accidents et densité de population", icon=":material/groups:") :
            st.write('''
            Nous remarquons que les départements les plus et moins accidentogènes sont des départements avec une forte et faible densité de population. \n
            Nos datasets ne mettent malheureusement pas en évidence le nombre d’usagers sur les routes permettant de vérifier cette hypothèse. Cependant, nous avons à notre disposition la densité de population par département, dans les tableaux ci-après.
            ''')
            st.write("\n")
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Densit%C3%A9%20de%20population%20par%20d%C3%A9partement.png", width = 400, caption="Les 10 départements avec la plus forte et la plus faible densité de population.")
            link_densite_population = "https://france.ousuisje.com/departements/classement/population.php"
            st.write("URL :", link_densite_population)
            st.write("\n")
            st.write('''
            Les départements 75, 92, 93, 94 font parties du top 6 des départements les plus accidentogènes et figurent dans le top 5 des départements avec la plus forte densité. \n
            Les départements 48, 23, 15 et 9 font parties du top 5 des départements les moins accidentogènes et figurent dans le top 8 des départements avec la plus faible densité de population. \n
            Cette simple lecture ne suffit toutefois pas à justifier la répartition de l’accidentologie puisque le département 13 est le second département le plus accidentogène mais est le 10ème département avec la plus forte densité de population, et à l’inverse, le département 04 est le 8ème département le moins accidentogène tout en étant le 3ème département avec la plus faible densité de population. \n
            Il y a des facteurs autres que la densité de population qui agissent sur le nombre d’accident.
            ''')

