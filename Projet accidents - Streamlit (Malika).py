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
caracteristiques = r"D:\\1- VS Code -Data Scientest\Fil rouge\caracteristiques.csv"
lieux = r"D:\1- VS Code -Data Scientest\Fil rouge\lieux.csv"
usagers = r"D:\1- VS Code -Data Scientest\Fil rouge\usagers.csv"
vehicules = r"D:\1- VS Code -Data Scientest\Fil rouge\vehicules.csv"

description_variables = r"D:\1- VS Code -Data Scientest\Fil rouge\accidents-routes-cda\Projets accidents - Description des variables.xlsx"

df_caracteristiques = pd.read_csv(caracteristiques, encoding = "ISO-8859-1", header=0, index_col=0)
df_lieux = pd.read_csv(lieux, encoding = "ISO-8859-1", header=0, index_col=0)
df_usagers= pd.read_csv(usagers, encoding = "ISO-8859-1", header=0, index_col=0)
df_vehicules= pd.read_csv(vehicules, encoding = "ISO-8859-1", header=0, index_col=0)
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

if page == pages[0] :
    st.write("\n\n\n\n\n")

    with st.expander("**A.** Contexte", icon=":material/contextual_token:") :
        st.write('''
            L’Observatoire national interministériel de la sécurité routière met à disposition chaque année depuis 2005, des Bases de données annuelles des accidents corporels de la circulation routière. \n
            Les bases de données, extraites du fichier BAAC, répertorient l'intégralité des accidents corporels de la circulation, intervenus durant une année précise en France métropolitaine, dans les départements d’Outre-mer (Guadeloupe, Guyane, Martinique, La Réunion et Mayotte depuis 2012) et dans les autres territoires d’outre-mer(Saint-Pierre-et-Miquelon, Saint-Barthélemy, Saint-Martin, Wallis-et-Futuna, Polynésie française et Nouvelle-Calédonie. 
        ''')
    st.write("\n\n\n")

    with st.expander("**B.** Objectifs", icon=":material/flag:") :
        st.write('''
            L’objectif de ce projet est de prédire la gravité des accidents routiers en France. \n
            La première étape sera d’opérer une exploration des différentes données présentes dans notre dataset. \n
            La seconde étape sera de mettre nos données en visualisation pour mieux comprendre le sujet étudié, ainsi que leur corrélation avec la variable cible de gravité. \n
            Il faudra ensuite appliquer les méthodes étudiées pendant notre cursus pour nettoyer le jeu de données. \n
            Nous créerons ensuite un modèle prédictif qui permettra d’anticiper la gravité des accidents en fonction des variables que l’on aura sélectionnés au préalable.

        ''')
    st.write("\n\n\n")

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
    
if page == pages[3] :
        st.write("\n\n\n\n\n")

        st.write("Après avoir effectué la visualisation, qui nous a permis d’analyser et de comprendre nos données, nous avons pu déterminer notre variable cible ainsi que les variables qui, selon nous, seront pertinentes pour notre Machine Learning.")
        st.write("Notre objectif est de sélectionner les variables qui ont un lien statistique avec la variable cible.")
        #st.write(':heavy_check_mark: Supprimer des doublons')
        st.write("\n")

        with st.expander("**A.** Supprimer des doublons", icon=":material/heap_snapshot_multiple:") :
            st.write('''
                Il y a 2858 lignes en doublons dans le dataset df_usagers. 
Nous les supprimons car ces doubles lignes vont démultiplier le nombre d’accident dans note dataset final.
            ''')
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20de%20doublon%20dans%20chaque%20dataset%20constituant%20notre%20jeu%20de%20donn%C3%A9e.png", width = 250, caption="Nombre de doublon dans chaque dataset constituant notre jeu de donnée.")


        with st.expander("**B.** Modifier le type de la variable « num_acc » du df_usagers", icon=":material/published_with_changes:") :
            st.write('''
                Le type d’origine de la variable « num_acc » est en float64, contrairement aux « num_acc » des autres dataset. Nous uniformisons la variable en modifiant son type en int64.
            ''')
        
        with st.expander("**C.** Fusionner les quatre datasets dans un seul DataFrame df_accidents", icon=":material/merge:") :
            with st.popover("Renommer les quatre colonnes « annee […] » de chaque dataset") :
                st.write('''
              Nous renommons les variables années de chaque dataset (« annee_caracteristiques », « annee_lieux », « annee_usagers » et « annee_vehicules »)..
           ''')
            with st.popover("Créer le DataFrame df_accidents en concaténant les quatre datasets") :
                st.write('''
             Nous effectuons la fusion en trois temps puisque les clés de jointure ne sont pas identiques)..
           ''')

        with st.expander("**D.** Supprimer les colonnes en doublon (annee_[…], num_veh_[…]) dans le df_accidents", icon=":material/delete:") :
            with st.popover("Supprimer les colonnes annee_[…] en doublon dans le df_accidents") :
                st.write('''
              Nous ne conserverons qu’une seule « année » pour notre dataset final. Nous supprimons les colonnes "annee_vehicules", "annee_usagers", "annee_lieux")..
           ''')
            with st.popover("Supprimer la colonne num_veh_[…] en doublon dans le df_accidents") :
                st.write('''
             La fusion a créé un « id_vehicule_y », nous supprimons cette colonne.)..
           ''')

        with st.expander("**E.** Renommer les colonnes annee_[…] et num_veh_[…] dans le df_accidents", icon=":material/signature:") :
            st.write('''
                Nous effectuons un renommage des colonnes dont le nom à changer ou que nous avons modifié avant la fusion.
« num_veh_x » redevient « num_veh » et « annee_caracteristiques » devient « annee ».
            ''')

        with st.expander("**F.** Supprimer les lignes du df_accidents pour lesquelles la colonne annee est supérieure ou égale à 2018", icon=":material/delete_history:") :
            st.write('''
                Comme indiqué précédemment, au vue des informations transmises, nous supprimons les données supérieur ou égale à l’année 2018.
            ''')

        with st.expander("**G.** Traitement sur les variables à conserver", icon=":material/construction:") :
            st.write('''
                En complément, du travail effectué sur chaque variable, pour notre analyse sur la temporalité, nous avons créé les variables suivantes : "jour_semaine", "date", "heure". 
Nous avons également supposé pertinent de créer une variable « age » de l'usager au moment de l'accident.
            ''')          
            #left_co, cent_co,last_co = st.columns(3)
            #with cent_co:
                #st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Travail%20effectu%C3%A9%20sur%20chaque%20variable.png", width = 350, caption="Pour chaque variable, nous avons effectué le travail suivant")
            st.markdown("<h6 style='text-align: center; color: grey;'>Pour chaque variable, nous avons effectué le travail suivant : </h6>", unsafe_allow_html=True)
            st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Travail%20effectu%C3%A9%20sur%20chaque%20variable.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
            st.write("\n\n\n\n\n")



        with st.expander("**H.** Variables conservées", icon=":material/dataset:") :
            st.write('''
                Pour donner suite à notre analyse lors de la visualisation, nous décidons de supprimer les colonnes non pertinentes pour les modèles dans le df_accidents. 
            ''')
            
            #st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Variables%20conserv%C3%A9s.png", width = 500, caption="Après la suppression notre dataset df_accidents est composé des colonnes suivantes ")
            st.markdown("<h6 style='text-align: center; color: grey;'>Après la suppression notre dataset df_accidents est composé des colonnes suivantes : </h6>", unsafe_allow_html=True)
            st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Variables%20conserv%C3%A9s.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
            st.write("\n\n\n\n\n")

#if page == pages[5] :




 # st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Densit%C3%A9%20de%20population%20par%20d%C3%A9partement.png", width = 400, caption="Les 10 départements avec la plus forte et la plus faible densité de population.")

