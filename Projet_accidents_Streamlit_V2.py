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
import requests
from io import BytesIO
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import traceback

 

#Chargement datasets en local
# @st.cache_data
# def charger_datasets():

#     df_caracteristiques=pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\caracteristiques.csv", encoding = "ISO-8859-1",index_col=0)
#     df_lieux = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\lieux.csv", encoding = "ISO-8859-1",index_col=0)
#     df_usagers = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\usagers.csv", encoding = "ISO-8859-1",index_col=0)
#     df_vehicules = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\vehicules.csv", encoding = "ISO-8859-1",index_col=0)
#     df_description_variables = pd.read_excel(r"C:\Users\adilb\Downloads\Projets accidents - Description des variables.xlsx")
#     df_accidents = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\5 - Rapport final et codes\df_accidents après pre-processing.csv", encoding = "ISO-8859-1",index_col=0)
#     return df_caracteristiques, df_lieux, df_usagers, df_vehicules, df_description_variables, df_accidents


try :
    st.set_page_config(layout="wide")
    st.title("Prédiction de la gravité des accidents :collision:")
    st.write("1- Lancement de l'app Streamlit... ✅")
    #Chargement datasets depuis GitHub
    @st.cache_data 
    def charger_datasets():
        df_caracteristiques = pd.read_csv(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Datas/accidents.csv").content), encoding = "ISO-8859-1", low_memory=False, header=0, index_col=0)
        df_lieux = pd.read_csv(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Datas/lieux.csv").content), encoding = "ISO-8859-1", low_memory=False, header=0, index_col=0)
        df_usagers= pd.read_csv(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Datas/usagers.csv").content), encoding = "ISO-8859-1", low_memory=False, header=0, index_col=0)
        df_vehicules= pd.read_csv(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Datas/vehicules.csv").content), encoding = "ISO-8859-1", low_memory=False, header=0, index_col=0)
        # df_description_variables = pd.read_excel(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Projets%20accidents%20-%20Description%20des%20variables.xlsx").content))
        df_description_variables = pd.read_excel("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Projets%20accidents%20-%20Description%20des%20variables.xlsx",engine="openpyxl")
        df_accidents = pd.read_csv(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Datas/accidents.csv").content))
        return df_caracteristiques, df_lieux, df_usagers, df_vehicules, df_description_variables, df_accidents

    df_caracteristiques, df_lieux, df_usagers, df_vehicules, df_description_variables, df_accidents = charger_datasets()
    st.write("2- Lancement de l'app Streamlit... ✅")
    #Configuration bandeau gauche de la page
        #Sommaire et navigation
    st.sidebar.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/DataScientest.png")
    st.sidebar.title("Sommaire")
    pages=["Introduction & Exploration", "Data Visualisation", "Pre-processing", "Modélisation", "Conclusion"]
    page=st.sidebar.radio("Aller vers", pages)
        #Auteurs
    st.sidebar.title("Auteurs :")
    st.sidebar.write("Adil Bouziane")
    st.sidebar.write("Malika Sadki")
    st.sidebar.write("Anouar Ouro-Sao")

    if page == pages[0] :

        st.write("\n\n\n\n\n")

        st.write("### :material/contextual_token: Contexte")
        
        st.write("\n\n")

        st.write('''
                L’Observatoire national interministériel de la sécurité routière met à disposition chaque année depuis 2005, des Bases de données annuelles des accidents corporels de la circulation routière. \n
                Les bases de données, extraites du fichier BAAC, répertorient l'intégralité des accidents corporels de la circulation, intervenus durant une année précise en France métropolitaine, dans les départements d’Outre-mer (Guadeloupe, Guyane, Martinique, La Réunion et Mayotte depuis 2012) et dans les autres territoires d’outre-mer(Saint-Pierre-et-Miquelon, Saint-Barthélemy, Saint-Martin, Wallis-et-Futuna, Polynésie française et Nouvelle-Calédonie. 
            ''')

        st.write("---")

        st.write("### :material/flag: Objectifs")
        
        st.write("\n\n")

        st.write('''
            L’objectif de ce projet est de prédire la gravité des accidents routiers en France. \n
            La première étape sera d’opérer une exploration des différentes données présentes dans notre dataset. \n
            La seconde étape sera de mettre nos données en visualisation pour mieux comprendre le sujet étudié, ainsi que leur corrélation avec la variable cible de gravité. \n
            Il faudra ensuite appliquer les méthodes étudiées pendant notre cursus pour nettoyer le jeu de données. \n
            Nous créerons ensuite un modèle prédictif qui permettra d’anticiper la gravité des accidents en fonction des variables que l’on aura sélectionnés au préalable.

        ''')

        st.write("---")

        st.write("### :material/edit_document: Description du BAAC")

        st.write("\n\n")

        with st.expander("Les forces de l'ordre renseignent les caractéristiques de chaque accident routier", icon=":material/local_police:") :
            st.write('''
                Tout usager impliqué dans un accident corporel de la circulation routière survenu sur le réseau routier ouvert à la circulation publique et impliquant au moins un véhicule doit en avertir les forces de l’ordre (gendarmerie nationale, sécurité publique, Préfecture de police de Paris, Compagnie Républicaine de Sécurité - article R 231-1 du code de la route). \n
                Ces dernières doivent remplir pour chaque accident corporel un Bulletin d’Analyse des Accidents Corporels (BAAC).
            ''')

        st.write("---")

        st.write("### :material/description: Description de la variable gravité")

        st.write("\n\n")

        with st.expander("La gravité de chaque accident est répertoriée, notée de 1 à 4 pour décrire 4 typologies de gravité", icon=":material/hotel_class:") :
            st.write('''
                * **1 = Indemne** \n
                * **2 = Tué** \n
                * **3 = Blessé grave** \n
                * **4 = Blessé léger** \n
            ''')
        with st.expander("Les forces de l'ordre interviennent sur des accidents corporels", icon=":material/activity_zone:") :
            st.write('''
                Un accident corporel (mortel et non mortel) de la circulation routière relevé par les forces de l’ordre : \n
                    • Implique au moins une victime, \n
                    • Survient sur une voie publique ou privée, ouverte à la circulation publique, \n
                    • Implique au moins un véhicule. \n
                Un accident corporel implique un certain nombre d’usagers. Parmi ceux-ci, on distingue : \n
                    • Les personnes indemnes : impliquées non décédées et dont l’état ne nécessite aucun soin médical du fait de l’accident, \n
                    • Les victimes : impliquées non indemnes. \n
                        o Les personnes tuées : personnes qui décèdent du fait de l’accident, sur le coup ou dans les trente jours qui suivent l’accident,\n
                        o Les personnes blessées : victimes non tuées.\n
                    • Les blessés dits « hospitalisés » : victimes hospitalisées plus de 24 heures, \n
                    • Les blessés légers : victimes ayant fait l'objet de soins médicaux mais n'ayant pas été admises 
                        comme patients à l'hôpital plus de 24 heures.
            ''')    
    

        st.write("---")

        st.write("### :material/description: Description de chaque dataset")

        st.write("\n\n")

        col1, col2, col3, col4 = st.columns(4, gap = "medium")

        with col1 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Logo%20Excel.png", width=100)
            st.write('''
                **Fichier 'caractristiques' :**\n
                Ce fichier contient les caractéristiques générales de chaque accident. Il 
                inclut des informations telles que la date, l'heure et le lieu de l'accident, ainsi que des détails sur les 
                circonstances particulières, comme les conditions météorologiques, l'éclairage et l'état de la route.
            ''')        
        with col2 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Logo%20Excel.png", width=100)
            st.write('''
                **Fichier 'lieux' :**\n
                Ce fichier contient des données sur les lieux où les accidents se sont produits. Il 
                comprend des informations sur le type de route (autoroute, route nationale, route départementale, 
                etc.), la configuration du carrefour, les zones urbaines ou rurales, et d'autres détails géographiques.
            ''')
        with col3 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Logo%20Excel.png", width=100)
            st.write('''
                **Fichier 'vehicules' :**\n
                Ce fichier contient des informations sur les véhicules impliqués dans chaque 
                accident. Il comprend des détails tels que le type de véhicule (voiture, moto, camion, etc.), la catégorie 
                du véhicule (véhicule léger, poids lourd, etc.), et des informations sur les dommages subis par les 
                véhicules.
            ''')
        with col4 :
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Logo%20Excel.png", width=100)
            st.write('''
                **Fichier 'usagers' :**\n
                Ce fichier contient des données sur les usagers impliqués dans chaque accident. Il 
                inclut des informations sur le type d'usager (conducteur, passager, piéton, cycliste, etc.), l'âge, le sexe 
                et le rôle de chaque usager dans l'accident.
            ''')             

        st.write("\n")

        st.write("**:material/search: Afficher les premières lignes de l'un des 4 datasets :**")

        checkbox_df_selected = st.selectbox("", ("df_caracteristiques","df_lieux","df_usagers","df_vehicules"), index=None, placeholder="Sélectionner un dataset", label_visibility="collapsed")
        st.write("\n")
        if checkbox_df_selected == "df_caracteristiques" :
            
            st.dataframe(df_caracteristiques.head(5))

        if checkbox_df_selected == "df_lieux" :
            st.dataframe(df_lieux.head(5))

        if checkbox_df_selected == "df_usagers" :
            st.dataframe(df_usagers.head(5))

        if checkbox_df_selected == "df_vehicules" :
            st.dataframe(df_vehicules.head(5))

        st.write("---")

        st.write("### :material/join_left: Jointure entre les 4 fichiers")

        st.write("\n\n")

        st.write("Notre phase d'exploration nous a permis de mettre la main sur la/le(s) clé(s) primaire(s) de chaque fichier :")
        st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Tables%20et%20jointures.png", caption="Jointure entre les 4 tables (fichiers CSV) mises à notre disposition.")

        st.write("**:material/info:** ***Il faudra être vigilant sur les clés primaires et faire un examen approfondi de la qualité de ces données. Les critères d'unicité, la similarité des formats de données, et le type de jointure choisi sont des critères primordiaux pour une fusion sans doublons ou pertes de données.***")

        st.write("---")

        st.write("### :material/explore: Pour pousser l'exploration encore plus loin...")

        st.write("\n\n")

        st.write("Notre phase d'exploration nous a permis de mettre la main sur la description de chacune des variables. Nous proposons une synthèse dans le tableau ci-après, avec la définition de chaque valeur.")
        st.dataframe(df_description_variables, hide_index=True)

    if page == pages[1] :

        #Créer 6 sections dans la page
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([":earth_americas: Localisation ", ":motorway: Route", ":partly_sunny: Condition extérieure", ":clock9: Temporalité", ":family: Humain", ":car: Véhicule"])

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

            st.write("---")

            #Titre 1
            st.write("### :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par type de localisation")

            st.write("\n\n")

            col1, col2 = st.columns(2, gap="medium")

            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d%E2%80%99accident%20par%20type%20de%20localisation.png", width = 550, caption="Nombre d'accident par type de location entre 2005 et 2017.")
            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d%E2%80%99accident%20par%20type%20de%20localisation%20par%20gravit%C3%A9.png", width = 500, caption="Nombre d'accident par type de localisation par gravité entre 2005 et 2017.")
            
            st.write("\n")

            with st.expander("**1.** Le nombre d'accident est deux fois plus élevé en agglomération qu'hors agglomération", icon=":material/location_city:") :
                st.write('''
                    Deux-tiers des accidents ont lieu en agglomération, un-tiers ont lieu hors-agglomération.
                ''')
            with st.expander("**2.** La gravité des accidents est plus forte hors agglomération", icon=":material/personal_injury:") :
                st.write('''
                    Pour les accidents hors-agglomération, il y a en tête du classement, les accidents « indemne », puis les accidents « blessé léger » et « blessé grave » avec respectivement, approximativement, 28.000, 22.000 et 21.500 accidents. \n
                    La répartition de la gravité des accidents en agglomération est différente. Les accidents « indemne » et « blessé grave » se dressent en tête du classement avec respectivement, approximativement, 62.000 et 56.000 accidents, alors que les accidents « blessé léger » et « tué » sont plus faible avec respectivement, approximativement, 23.000 et 2.000 accidents. \n
                    L’écart entre les types de gravité d’accidents est beaucoup plus faible en nombre lorsque les accidents sont situés hors-agglomération. Le nombre d’accidents de type « tué » est plus important (environ 4.000) hors-agglomération, qu’en agglomération (environ 2.000), alors que le nombre d’accident hors￾agglomération est deux fois moins important que le nombre d’accident en agglomération. \n
                    En conclusion, il y a des éléments derrière le type d’agglomération qui ont l’air d’avoir un impact sur la gravité des accidents.
                ''')

        #Section "Route"
        with tab2 :
            st.write("\n\n\n\n\n\n\n\n\n\n")

            #Titre 1
            st.write("### :material/road: :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par localisation de l'accident sur la route")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="situ")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20localisation%20accident%20sur%20la%20route%20par%20gravit%C3%A9.png", caption="Nombre d'accident par localisation sur la route par gravité entre 2005 et 2017.")
            
            st.write("\n")

            with st.expander("**1.** Les accidents ont lieu en majorité sur la chaussée, puis nettement moins sur l'accotement et le trottoir", icon=":material/road:") :
                st.write('''
                    Un peu plus de 500.000 accidents ont eu lieu sur des routes communales, environ 320.000 sur des routes départementales, 80.000 sur des routes nationales et 80.000 sur des autoroutes. \n
                    Des accidentsont eu lieu sur d’autres catégories de route, mais cela reste à la marge comparé aux 4 premières catégories. \n
                    Cela confirme la tendance que l'on a constaté précédemment lors de l'analyse de la variable « agg », c'est-à-dire que le plus grand nombre d'accidents a lieu sur des routes de ville.
                ''')
            with st.expander("**2.** Lorsque les accidents se produisent sur l'accotement, la majorité des usagers connaissent des blessures grave, contrairement aux accidents sur chaussée qui enregistrent une majorité d'usagers indemnes", icon=":material/personal_injury:") :
                st.write('''
                    Nous remarquons une distribution similaire a ce qu’on nous avons pu constaté auparavant sur les 
                    types de gravité sur la valeur « En agglo » de la variable « agg », à savoir une majorité d’accident 
                    « indemne », puis légèrement en dessous les accidents « blessé léger ». Les accidents « blessé grave » 
                    représente un peu moins de 50% des accidents « indemne », et un faible nombre d’accident « tué » au 
                    regard du nombre d’accident global. \n
                    Il est intéressant de noter que les accidents dont le point de choc initial s’est produit sur l’accotement 
                    font drastiquement chuter la gravité « indemne » comparé à ce que nous avons constaté jusque-là. 
                    Dans ce cas, c’est la gravité « blessé grave » qui arrive en première position, suivi de la gravité « blessé 
                    léger ». \n
                    Le constat est le même lorsque le point de choc initial se produit sur le trottoir, ou qu’aucun point de 
                    choc initial n’a été relevé lors de l’accident. 
                ''')
    
            st.write("---")

            #Titre 1
            st.write("### :material/road: :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par catégorie de route")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="catr")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nbre%20d'accident%20par%20Gravit%C3%A9%20par%20Cat%C3%A9gorie%20de%20Route.png", caption="Nombre d'accident par catégorie de route par gravité entre 2005 et 2017.")
            
            st.write("\n")

            with st.expander("**1.** Les accidents ont lieu en majorité sur les voies communales, puis sur les routes départementales, et enfin sur les routes nationales et autoroutes", icon=":material/road:") :
                st.write('''
                    Un peu plus de 500.000 accidents ont eu lieu sur des routes communales, environ 320.000 sur des routes départementales, 80.000 sur des routes nationales et 80.000 sur des autoroutes. \n
                    Des accidentsont eu lieu sur d’autres catégories de route, mais cela reste à la marge comparé aux 4 premières catégories. \n
                    Cela confirme la tendance que l'on a constaté précédemment lors de l'analyse de la variable « agg », c'est-à-dire que le plus grand nombre d'accidents a lieu sur des routes de ville.
                ''')
            with st.expander("**2.** La gravité des accidents est plus forte sur les routes départementales et nationales", icon=":material/personal_injury:") :
                st.write('''
                Les routes départementales présente un nombre d'accidents avec une gravité « blessé grave » en 2èmeposition juste après le nombre d'accidents avec une gravité « indemne ». Les autres catégories de routes ont à la seconde position la gravité « blessé léger ». \n
                De manière générale, les routes départementales, nationales et autoroutes ont un écart plus faible entre les gravités « blessé léger », « blessé grave » et « indemne » en comparaison aux voiescommunales. \n
                Les routes départementales, nationales, et autoroutes sont spécifiques aux routes hors￾agglomération, et nous conforte donc dans la conclusion rendue précédemment lors de l'analyse de la variable « agg ». \n
                Il faut investiguer sur les caractéristiques de ces routes qui ont un impact indéniable sur la gravité des 
                accidents.
                ''')

        #Section "Condition extérieure"
        with tab3 :
            st.write("\n\n\n\n\n\n\n\n\n\n")

            #Titre 1
            st.write("### :material/car_crash: :material/personal_injury: Accidents par type de collision")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="col")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20collision%20par%20gravit%C3%A9.png", caption="Nombre d'accident par collision par gravité entre 2005 et 2017.")
            
            st.write("\n")

            with st.expander("**1.** La majorité des accidents concernent des collisions multiples entre 3 véhicules et plus, et des collisions entre 2 véhicules par l'arrière", icon=":material/car_crash:") :
                st.write('''
                    La majorité des accidents avec collision concernent des collisions entre deux véhicules, par l’arrière 
                    (29% des accidents), et des collisions entre trois véhicules et plus avec multiples collisions (environ 
                    35% des accidents). 
                ''')
            with st.expander("**2.** La proportion d'usager indemne est plus faible dans certains cas de collision", icon=":material/personal_injury:") :
                st.write('''
                    Lorsque les accidents sont issues d’une collision de 3 véhicules avec multiples collisions, et d’une 
                    collision de 2 véhicules en frontale, les gravités « indemne » et « blessé léger » sont au coude à coude, 
                    voir même égales dans le 1er cas. \n
                    Il y a tout de même un cas « Non-renseigné » où la gravité « blessé grave » est en 1ère position, devant 
                    « blessé léger », et « indemne ». 
                ''')
    
            st.write("---")

            #Titre 1
            st.write("### :material/add_triangle: :material/car_crash: :material/personal_injury: Accidents par obstacle mobile heurté")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="obsm")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20obstacle%20mobile%20heurt%C3%A9%20par%20gravit%C3%A9.png", caption="Nombre d'accident par obstacle mobile heurté par gravité entre 2005 et 2017.")
            
            st.write("\n")

            st.write('''
                Les gravités « blessé léger » et « blessé grave » sont au-dessus de la gravité « indemne » dans le cas où 
                aucun obstacle n’est heurté. 
            ''')

        #Section "Temporalité"
        with tab4 :
            st.write("\n\n\n\n\n\n\n\n\n\n")

            #Titre 1
            st.write("### :material/today: :material/car_crash: :material/personal_injury: Accidents par jour de la semaine")

            st.write("\n\n")

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20jour%20de%20la%20semaine%20par%20gravit%C3%A9.png", caption="Nombre d'accident par jour de la semaine par gravité entre 2005 et 2017.")
            
            st.write("\n")

            st.write('''
                Le vendredi est le jour le plus accidentogène, à l’inverse du dimanche qui est la journée la moins 
                accidentogène. \n
                La gravité "Blessé léger" et "Indemne" est au coude à coude sur la journée du mardi.  
            ''')

            st.write("---")

            #Titre 1
            st.write("### :material/schedule: :material/car_crash: :material/personal_injury: Accidents par heure")

            st.write("\n\n")

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20heure%20par%20gravit%C3%A9.png", caption="Nombre d'accident par heure par gravité entre 2005 et 2017.")
            
            st.write("\n")

            with st.expander("**1.** Le nombre d'accident est plus fort lorsque la fréquentation est plus élevée", icon=":material/work_history:") :
                st.write('''
                    L’analyse met en évidence que le nombre d'accident en début de journée est plus élevé à 8h, et en fin 
                    de journée le nombre d'accident est plus élevé entre 16h et 18h. Le nombre d’accident est beaucoup 
                    plus faible entre 23h et 6h. \n
                    Ceci est indéniablement une conséquence directe de la fréquentation de la 
                    route qui est plus élevée sur ces heures en raison des trajets domicile-travail et travail-domicile.  
                ''')
            with st.expander("**2.** La gravité des accidents est plus forte la nuit", icon=":material/nightlight:") :
                st.write('''
                    A partir de 21h, et jusqu’à 5h, le nombre d’accident en gravité « blessé léger » est supérieur au 
                    nombre d’accident en gravité « indemne ». 
                ''')

        #Section "Humain"
        with tab5 :
            st.write("\n\n\n\n\n\n\n\n\n\n")

            #Titre 1
            st.write("### :material/search_hands_free: :material/car_crash: :material/personal_injury: Accidents par manoeuvre")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="manv")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            col1, col2 = st.columns(2, gap="medium", vertical_alignment = "center")

            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20manoeuvre%20par%20gravit%C3%A9.png", width = 550, caption="Nombre d'accident par type de location entre 2005 et 2017.")

            with col2 :
            
                st.write("\n")

                with st.expander("**1.** Le majorité des accidents a lieu sans intention de changement de direction des usagers", icon=":material/search_hands_free:") :
                    st.write('''
                        La majorité des accidents se produisent sans intention de 
                        changement de direction (environ 810.000). En seconde 
                        position nous retrouvons des accidents qui se produisent 
                        dans la même file et le même sens de direction (environ 
                        210.000). \n
                        Sous la barre des 200.000 accidents, nous retrouvons 
                        dans l’ordre décroissant les accidents ayant eu lieu 
                        lorsque le véhicule tournait à gauche (environ 180.000), 
                        lorsque le véhicule se déportait à gauche (environ 
                        80.000), et lorsqu’il est arrêté, sans stationner (environ 
                        50.000). \n
                        On dénombre tout de même 150.000 lignes de véhicules 
                        dont la manœuvre est inconnue.  
                    ''')
                with st.expander("**2.** Les accidents lors de manoeuvre de déportement, ou de dépassement, à gauche sont à l'origine d'une gravité plus forte", icon=":material/personal_injury:") :
                    st.write('''
                    Sur les accidents dont la manœuvre 
                    principale est « Sans changement de 
                    direction », la gravité « blessé léger » est 
                    en première position (environ 380.000 
                    accidents), devant la gravité « indemne » 
                    (environ 360.000 accidents). \n
                    Lorsque les accidents ont lieu dans la 
                    même file et dans le même sens, la 
                    gravité « indemne » est en première 
                    position (environ 120.000 accidents), 
                    devant la gravité « blessé léger » (environ 
                    58.000 accidents). \n
                    L’écart entre les accidents « indemne » 
                    (environ 120.000), en première position, 
                    et « blessé léger » (environ 80.000), en 
                    deuxième position, est encore plus grand 
                    lorsque les accidents ont eu lieu lorsque 
                    les véhicules tournaient à gauche. \n
                    Lors des manœuvres « Déporté – A 
                    gauche » et « Dépassant – A gauche » le 
                    nombre d’accidents « indemne » se 
                    retrouve en 3ème position. \n
                    Lorsque les véhicules se sont déportés à 
                    gauche, la gravité majoritaire est la 
                    gravité « blessé grave », suivi de « blessé 
                    léger ». A l’inverse, lorsque les véhicules 
                    dépassaient par la gauche, la gravité 
                    majoritaire est la gravité « blessé léger », 
                    suivi de « blessé grave ». \n
                    Nous constatons que la manœuvre 
                    effectué par les véhicules juste avant 
                    l’accident a un impact sur la gravité de 
                    l’accident. 
                    ''')

            st.write("---")

            #Titre 1
            st.write("### :material/category: :material/car_crash: :material/personal_injury: Accidents par catégorie d'usager")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="catu")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20cat%C3%A9gorie%20d'usager%20par%20gravit%C3%A9.png", caption="Nombre d'accident par catégorie d'usager par gravité entre 2005 et 2017.")

            st.write("\n")

            st.write("La très grande majorité des accidents concerne, sans surprise, les conducteurs.")

            with st.expander("Les passagers ont plus de chance de ressortir d'un accident avec des blessures légères et les piétons n'ont quasiment aucune chance de ressortir indemne d'un accident", icon=":material/category:") :
                st.write('''
                    La gravité « indemne » diminue significativement en passant de conducteur, à passager, pour finir par 
                    être quasiment absente lorsque la catégorie d’usager est « piéton ».\n
                    La gravité « indemne » est en 1ère position lorsqu’il s’agit des conducteurs, passe en 2ème position 
                    lorsqu’il s’agit des passagers, après la gravité « blessé léger ». \n
                    Les piétons sont sujets à une gravité « blessé léger » en majorité, puis en gravité « blessé grave » et en 
                    gravité « tué ». C’est la première dimension que l’on analyse où la gravité « tué » est devant la gravité 
                    « indemne ».
                ''')

            st.write("---")

            #Titre 1
            st.write("### :material/airline_seat_recline_normal: :material/car_crash: :material/personal_injury: Accidents par position de l'usager dans le véhicule")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="place")|(df_description_variables["Variables"]=="grav")], hide_index=True)

            with st.popover("Schéma d'identification de la position de l'usager dans le véhicule", icon=":material/radar:") :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Position%20de%20l'usager%20dans%20le%20v%C3%A9hicule.png")

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20position%20de%20l'usager%20par%20gravit%C3%A9.png", caption="Nombre d'accident par position de l'usager par gravité entre 2005 et 2017.")

            st.write("\n")

            st.write("La très grande majorité des accidents concerne, sans surprise, les conducteurs.")

            with st.expander("Les conducteurs ont plus de chance de ressortir indemne d'un accident et les passagers aux extrémités d'un véhicule ont plus de chance d'en ressortir avec des blessures légères qu'indemne", icon=":material/category:") :
                st.write('''
                    La gravité « indemne » est en 2ème position lorsque les usagers sont à des places autre que celle du 
                    conducteur. \n
                    On constate que lorsque les usagers sont des passagers placés aux extrémités du véhicule la gravité 
                    d’un accident est plus élevé puisque « blessé léger » prend la 1ère position.
                ''')

            st.write("---")

            #Titre 1
            st.write("### :material/cake: :material/car_crash: :material/personal_injury: Accidents par âge de l'usager")

            st.write("\n\n")

            st.write("La variable « age » n’existe pas dans notre dataset, mais nous pouvons la calculer grâce à l’année de naissance de l’usager (« an_naiss ») et l’année de l’accident (« année ») : **age = année – an_naiss**.")        
            
            st.write("\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="grav")], hide_index=True)

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20age.png", caption="Nombre d'accident par âge de l'usager entre 2005 et 2017.")
            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20tranche%20d'age%20par%20gravit%C3%A9.png", caption="Nombre d'accident par tranche d'âge de l'usager par gravité entre 2005 et 2017.")

            st.write("\n")

            with st.expander("Les jeunes sont majoritaires dans les accidents de la route avec un pic d'accident pour les 19 ans", icon=":material/cake:") :
                st.write('''
                Il est indéniable que les jeunes sont les plus concernés par les accidents. Le pic d’accident se situe 
                pour les usagers de 19 ans. A partir de la vingtaine, les accidents décroissent fortement jusqu’à la 
                quarantaine, pour décroitre timidement jusqu’à la cinquantaine, avant de reprendre une forte 
                décroissance.
                ''')
            with st.expander("Les fragilités physiques des personnes âgées et les comportements routiers spécifiques aux jeunes personnes ont une implication dans la gravité des accidents auxquels ils sont soumis", icon=":material/personal_injury:") :
                st.write('''
                Il est indéniable que les jeunes sont les plus concernés par les accidents. Le pic d’accident se situe 
                pour les usagers de 19 ans. A partir de la vingtaine, les accidents décroissent fortement jusqu’à la 
                quarantaine, pour décroitre timidement jusqu’à la cinquantaine, avant de reprendre une forte 
                décroissance. \n
                De 0 à 17 ans les gravités « blessé léger » et « blessé grave » sont les deux gravités majoritaires. Pour 
                la tranche 18-24 ans le nombre d’accident augmente fortement, avec un nombre de gravité « blessé 
                léger » au-dessus de la gravité « indemne ». \n
                Plus on avance dans les tranches d’âge et plus la gravité « indemne » en première position diminue, au 
                profit d’un rattrapage des gravités « blessé léger » et « blessé grave », jusqu’à la tranche d’âge 81-90 
                ans pour laquelle la gravité « blessé grave » est la plus présente. \n
                L’âge est corrélé à la gravité des accidents. Cette variable permet de mettre en évidence deux 
                dimensions qui ne sont pas présentes dans notre dataset : les fragilités physiques (pour les personnes 
                âgées) ou des comportements routiers spécifiques (manque de vigilance, excès de confiance, etc…) 
                propres à des tranches d’âges spécifiques.
                ''')

        with tab6 :

            st.write("\n\n\n\n\n\n\n\n\n\n")

            #Titre 1
            st.write("### :material/destruction: :material/car_crash: :material/personal_injury: Accidents par point de choc initial")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="choc")|(df_description_variables["Variables"]=="grav")], hide_index=True)  

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20choc%20par%20gravit%C3%A9.png", caption="Nombre d'accident par point de choc intial par gravité entre 2005 et 2017.")

            st.write("\n")

            with st.expander("Les blessés légers sont majoritaires lorsqu'il n'y a aucun point de choc, ou un point de choc à l'avant, ou des tonneaux. Pour ce dernier cas, les usagers ont même plus de chance d'en ressortir avec des blessures légères ou graves", icon=":material/destruction:") :
                st.write('''
                    Dans toutes les configurations de point de choc initial, la répartition des gravités se fait dans l’ordre 
                    « indemne », « blessé léger », « blessé grave », et « tué », sauf dans le cas du point de choc initial 
                    « Avant » et « Aucun », où la gravité « blessé léger » est la plus présente, devant la gravité 
                    « indemne ». Autre exception, lorsqu’il y a des chocs multiples dans le cas de tonneaux, les gravités 
                    dominantes sont « blessé léger » et « blessé grave ». \n
                    Dans le cas des accidents où le point de choc initial est à l’arrière, sur le côté droit ou sur le côté 
                    gauche, la gravité « blessé léger » est presque au même niveau que la gravité « indemne ». \n
                    Il y a une réelle corrélation entre les points de chocs et la gravité d’un accident.
                ''')

            st.write("---")

            #Titre 1
            st.write("### :material/transportation: :material/car_crash: :material/personal_injury: Accidents par catégorie de véhicule")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="catv")|(df_description_variables["Variables"]=="grav")], hide_index=True)  

            st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20d'accident%20par%20cat%C3%A9gorie%20de%20v%C3%A9hicule%20par%20gravit%C3%A9.png", caption="Nombre d'accident par catégorie de véhicule par gravité entre 2005 et 2017.")

            st.write("\n")

            with st.expander("Les usagers de deux-roues connaissent des accidents avec des gravités plus fortes que les autres types de véhicules", icon=":material/two_wheeler:") :
                st.write('''
                    La catégorie de véhicule 7, représentant les VL seuls, est la plus impliquée dans les accidents de la 
                    route, et présente une gravité « indemne » prépondérante. \n
                    Cependant, le graphique met en évidence que les catégories de véhicule 30 à 34 présentent une 
                    gravité « indemne » très faible par rapport aux autres gravités. Elle est presque absente sur les 
                    catégories 31 et 34. Sur la catégorie de véhicule 33, les gravités « blessé léger » et « blessé grave » 
                    sont pratiquement au même niveau, tout comme les gravités « indemne » et « tué ». Ces catégories 
                    présentent une particularité intéressante : ce sont tous des scooters/motocyclettes supérieures ou 
                    égales à 50cm3. \n
                    L’analyse de ce graphique met en avant un critère qui parait assez évident : les usagers de deux roues 
                    sont sujets à des accidents plus grave que les autres types de véhicules, très certainement du fait du 
                    peu de carénage qui les protègent.
                ''')

    if page == pages[2] :

        st.write("\n\n\n\n\n")

        st.write("Après avoir effectué la visualisation, qui nous a permis d’analyser et de comprendre nos données, nous avons pu déterminer notre variable cible ainsi que les variables qui, selon nous, seront pertinentes pour notre Machine Learning.")
        st.write("Notre objectif est de sélectionner les variables qui ont un lien statistique avec la variable cible.")
        #st.write(':heavy_check_mark: Supprimer des doublons')
        st.write("\n")

        with st.expander("**1.** Supprimer des doublons", icon=":material/heap_snapshot_multiple:") :
            st.write('''
                Il y a 2858 lignes en doublons dans le dataset df_usagers. 
                Nous les supprimons car ces doubles lignes vont démultiplier le nombre d’accident dans note dataset final.
            ''')
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Nombre%20de%20doublon%20dans%20chaque%20dataset%20constituant%20notre%20jeu%20de%20donn%C3%A9e.png", width = 250, caption="Nombre de doublon dans chaque dataset constituant notre jeu de donnée.")


        with st.expander("**2.** Modifier le type de la variable « num_acc » du df_usagers", icon=":material/published_with_changes:") :
            st.write('''
                Le type d’origine de la variable « num_acc » est en float64, contrairement aux « num_acc » des autres dataset. Nous uniformisons la variable en modifiant son type en int64.
            ''')
        
        with st.expander("**3.** Fusionner les quatre datasets dans un seul DataFrame df_accidents", icon=":material/merge:") :
            with st.popover("Renommer les quatre colonnes « annee […] » de chaque dataset") :
                st.write('''
                    Nous renommons les variables années de chaque dataset (« annee_caracteristiques », « annee_lieux », « annee_usagers » et « annee_vehicules »)..
                ''')
            with st.popover("Créer le DataFrame df_accidents en concaténant les quatre datasets") :
                st.write('''
                    Nous effectuons la fusion en trois temps puisque les clés de jointure ne sont pas identiques).
                ''')

        with st.expander("**4.** Supprimer les colonnes en doublon (annee_[…], num_veh_[…]) dans le df_accidents", icon=":material/delete:") :
            with st.popover("Supprimer les colonnes annee_[…] en doublon dans le df_accidents") :
                st.write('''
                    Nous ne conserverons qu’une seule « année » pour notre dataset final. Nous supprimons les colonnes "annee_vehicules", "annee_usagers", "annee_lieux").
                ''')
            with st.popover("Supprimer la colonne num_veh_[…] en doublon dans le df_accidents") :
                st.write('''
                    La fusion a créé un « id_vehicule_y », nous supprimons cette colonne.)..
                ''')

        with st.expander("**5.** Renommer les colonnes annee_[…] et num_veh_[…] dans le df_accidents", icon=":material/signature:") :
            st.write('''
                Nous effectuons un renommage des colonnes dont le nom à changer ou que nous avons modifié avant la fusion.
                « num_veh_x » redevient « num_veh » et « annee_caracteristiques » devient « annee ».
            ''')

        with st.expander("**6.** Supprimer les lignes du df_accidents pour lesquelles la colonne annee est supérieure ou égale à 2018", icon=":material/delete_history:") :
            st.write('''
                Lors du lancement de notre projet Fil Rouge, nous avons pris la décision de prendre la base de 
                données contenant la totalité des informations de 2005 à 2021 pour chacun des 4 fichiers 
                (Caractéristiques – Lieux – Véhicules – Usagers) au format CSV. \n
                Nous avons fait le choix de conserver les données sur les 13 premières années (2005 à 2017) pour 
                chaque fichier puisque le document de description des bases de données annuelles des accidents, mis 
                à notre disposition par L' Observatoire national interministériel de la sécurité routière, nous apporte 
                cet avertissement :\n
                « *Les données sur la qualification de blessé hospitalisé depuis l’année 2018 ne pouvant être comparées 
                aux années précédentes suite à des modifications de process de saisie des forces de l’ordre. L’indicateur 
                « blessé hospitalisé » n’est plus labellisé par l’autorité de la statistique publique depuis 2019.
                A partir des données de 2021, les usagers en fuite ont été rajoutés, cela entraîne des manques 
                d’informations sur ces derniers, notamment le sexe, l’âge, voire la gravité des blessures 
                (indemne,blessé léger ou blessé hospitalisé).* ». \n
                Nous supprimons donc toutes les lignes dont les accidents sont enregistrés à partir de 2018.
            ''')

        with st.expander("**7.** Traitement sur les variables à conserver", icon=":material/construction:") :
            st.write('''
                En complément, du travail effectué sur chaque variable, pour notre analyse sur la temporalité, nous avons créé les variables suivantes : "jour_semaine", "date", "heure". 
                Nous avons également supposé pertinent de créer une variable « age » de l'usager au moment de l'accident.
            ''')
            st.markdown("<h6 style='text-align: center; color: grey;'>Pour chaque variable, nous avons effectué le travail suivant : </h6>", unsafe_allow_html=True)
            st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Travail%20effectu%C3%A9%20sur%20chaque%20variable.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
            st.write("\n\n\n\n\n")



        with st.expander("**8.** Variables conservées", icon=":material/dataset:") :
            
            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="mois")|(df_description_variables["Variables"]=="jour")|(df_description_variables["Variables"]=="lum")|(df_description_variables["Variables"]=="agg")|(df_description_variables["Variables"]=="int")|(df_description_variables["Variables"]=="atm")|(df_description_variables["Variables"]=="col")|(df_description_variables["Variables"]=="dep")|(df_description_variables["Variables"]=="annee")|(df_description_variables["Variables"]=="catr")|(df_description_variables["Variables"]=="circ")|(df_description_variables["Variables"]=="plan")|(df_description_variables["Variables"]=="surf")|(df_description_variables["Variables"]=="situ")|(df_description_variables["Variables"]=="place")|(df_description_variables["Variables"]=="catu")|(df_description_variables["Variables"]=="grav")|(df_description_variables["Variables"]=="catv")|(df_description_variables["Variables"]=="obsm")|(df_description_variables["Variables"]=="choc")|(df_description_variables["Variables"]=="manv")], hide_index=True)

            st.write('''
                Pour donner suite à notre analyse lors de la visualisation, nous décidons de supprimer les colonnes non pertinentes pour les modèles dans le df_accidents. 
            ''')
            st.markdown("<h6 style='text-align: center; color: grey;'>Après la suppression notre dataset df_accidents est composé des colonnes suivantes : </h6>", unsafe_allow_html=True)
            st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Variables%20conserv%C3%A9s.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
            st.write("\n\n\n\n\n")

    if page == pages[3] :

        #Créer 5 sections dans la page
        tab1, tab2, tab3, tab4, tab5 = st.tabs([":bulb: Overview",":deciduous_tree: Random Forest Classifier", ":deciduous_tree: Decision Tree Classifier", ":deciduous_tree: CatBoost Classifier", ":arrow_forward: Démo"])

        with tab1 :
            st.write("\n\n\n\n\n\n\n\n\n\n")

            st.write("### :material/visibility: Aperçu des premières lignes de notre dataset final")

            st.write("\n\n")

            with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables[(df_description_variables["Variables"]=="mois")|(df_description_variables["Variables"]=="jour")|(df_description_variables["Variables"]=="lum")|(df_description_variables["Variables"]=="agg")|(df_description_variables["Variables"]=="int")|(df_description_variables["Variables"]=="atm")|(df_description_variables["Variables"]=="col")|(df_description_variables["Variables"]=="dep")|(df_description_variables["Variables"]=="annee")|(df_description_variables["Variables"]=="catr")|(df_description_variables["Variables"]=="circ")|(df_description_variables["Variables"]=="plan")|(df_description_variables["Variables"]=="surf")|(df_description_variables["Variables"]=="situ")|(df_description_variables["Variables"]=="place")|(df_description_variables["Variables"]=="catu")|(df_description_variables["Variables"]=="grav")|(df_description_variables["Variables"]=="catv")|(df_description_variables["Variables"]=="obsm")|(df_description_variables["Variables"]=="choc")|(df_description_variables["Variables"]=="manv")], hide_index=True)

            st.dataframe(df_accidents.head())

            st.write("---")

            st.write("### :material/flag: Objectifs")

            st.write("\n\n")

            with st.expander("Pour rappel, notre but est de prédire la gravité des accidents de la route en France, grâce à la variable 'grav'", icon=":material/hotel_class:") :
                st.write('''
                    * **1 = Indemne** \n
                    * **2 = Tué** \n
                    * **3 = Blessé grave** \n
                    * **4 = Blessé léger** \n
                ''')

            st.write("---")

            st.write("### :material/question_mark: Quel(s) modèle(s) de Machine Learning utiliser ?")

            with st.expander("Généralité sur le Machine Learning", icon=":material/sticky_note_2:") :
                st.write('''
                    Grâce au Machine Learning, nous pouvons prédire la valeur d'une variable cible à partir de variables explicatives.\n
                    Les variables explicatives ont été déterminées précédemment, grâce à la partie Data Visualisation. Sont considérées comme variables explicatives toutes les variables ayant une corrélation avec la variable cible. Il appartient au modélisateur de définir ce qu'il considère comme variables explicatives ou non. \n
                    Les différentes valeurs prises par la variable cible sont ce qu'on appelle des classes.
                ''')
            with st.expander("Sélection des modèles", icon=":material/done_outline:") :  
                st.write('''
                        Nous sommes dans un problème de classification, nous choisirons le modèle adapté pour mettre en place notre modèle de Machine Learning.\n
                        L'objectif de la classification est donc de prédire la classe d'une observation à partir de ses variables explicatives. \n
                        Nous utiliserons donc 3 modèles de Machine Learning, adapté à cette problématique, que nous allons mettre en compétition pour définir le plus performant : \n
                        * **Random Forest Classifier** \n
                        * **Decision Tree Classifier** \n
                        * **Cat Boost Classifier** \n
                ''')

            st.write("---")

            st.write("### :material/dictionary: Quelques définitions importantes")

            with st.expander("Mécanisme du Machine Learning", icon=":material/manufacturing:") :
                st.write('''
                    * **Le jeu d'entrainement** sert à entraîner le modèle de classification,c'est-à-dire à trouver les paramètres du modèle qui séparent au mieux les classes. Le modèle va apprendre et s'entraîner \n
                    * **Le jeu de test** sert à évaluer le modèle sur des données qu'il n'a jamais vues. Cette étape nous permettra d'évaluer la capacité du modèle à se généraliser \n
                    * **L’overfitting** est un problème en apprentissage automatique où un modèle apprend trop bien les données d'entraînement, au point de capturer le bruit et les détails inutiles. Cela le rend moins performant sur de nouvelles données, car il ne généralise pas bien \n
                ''')
            with st.expander("Métriques pour évaluer la performance de nos modèles de Machine Learning", icon=":material/readiness_score:") :  
                st.write('''
                        * **Score** : Calcule une métrique de performance du modèle en comparant les vraies valeurs de la variable cible à la prédiction \n
                        * **Recall** : Mesure la capacité du modèle à détecter correctement toutes les instances positives \n
                        * **F1-score** : Utile pour les problèmes de classification binaire car elle prend en compte à la fois la précision et le rappel pour calculer un score global \n
                        * **Accuracy** : Métrique d’évaluation du pourcentage de prédictions correctes faites par un modèle \n
                ''')

            st.write("---")

            st.write("### :material/arrow_split: Préparation de notre dataset final pour le Machine Learning")

            st.write('''
                Notre jeu de donnée a été séparé en deux DataFrame distincts :\n
                * **X** : Contient les variables explicatives\n
                * **y** : Contient la variable cible\n          
                Pour chaque modèle que nous allons lancer, nous séparons le jeu de donnée en un jeu de test pour **20%**  et un jeu d’entrainement pour **80%** selon la formule suivante :
            ''')
            st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)", language="python")
                
        with tab2 :

            st.write("\n\n\n\n\n\n\n\n\n\n")

            st.write("### :material/dictionary: Définition")

            st.write("\n\n")

            st.write('''
                Le **RandomForestClassifier** est un algorithme d'apprentissage automatique utilisé pour la classification. \n
                Il appartient à la famille des **forêts aléatoires (Random Forest)**, qui est une méthode d'ensemble basée 
                sur plusieurs **arbres de décision**.
            ''')

            st.write("---")

            st.write("### :material/manufacturing: Fonctionnement du modèle")

            st.write("\n\n")

            st.write('''
                * Il permet de créer plusieurs arbres de décision sur des sous-échantillons aléatoire de donnée\n
                * Chaque arbre fait une prédiction\n
                * La classe finale est déterminée par un **vote majoritaire** des arbres      
            ''')

            st.write("---")

            st.write("### :material/add_circle: Avantages du modèle")

            st.write("\n\n")

            st.write('''
                * Il est plus précis et plus robuste qu’un seul arbre de décision\n
                * Il est moins sensible au surapprentissage (overfitting)\n
                * Il gère bien les données bruitées et les variables non pertinentes\n
            ''')

            st.write("---")

            st.write("### :material/do_not_disturb_on: Inconvénient du modèle")

            st.write("\n\n")

            st.write('''
                * Le modèle peut être difficile à interpréter\n
            ''')

            st.write("---")

            st.write("### :material/labs: Résultats du modèle et analyses")

            st.write("\n\n")

            col1, col2 = st.columns(2, gap="large")

            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_Random_Forest_Sans_Hyperarametre.png", caption="Random Forest sans hyper paramètres")
                
                st.write("\n")

                st.write('''
                    Comme attendu, nous constatons que notre modèle génère un overfitting très important. Nous n’utiliserons pas ce modèle.
                ''')

            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_Random_Forest_Avec_Hyperparametre.png", caption="Random Forest avec hyper paramètres (max_depth=8)")
                
                st.write("\n")

                with st.expander("Premier niveau de performance atteint avec la réduction de l'overfitting", icon=":material/settings:") :
                    st.write('''
                        L’ajout d’un hyper paramètre, rend notre modèle plus performant en réduisant l’overfitting. La différence de score entre notre jeu de test et de notre jeu d’entrainement est considérablement réduite. Le résultat, qui s’élève à 0,62 (jeu d’entrainement et jeu de test), n’est pas aussi bon que le premier modèle car sur le premier nous avons un résultat de 0,99 (jeu d’entrainement) et 0,65 (jeu de test).\n
                        Le modèle peut malgré tout être considéré comme plus performant que la première version car il se trompe moins dans la prédiction.
                    ''')

        with tab3 :

            st.write("\n\n\n\n\n\n\n\n\n\n")

            st.write("### :material/dictionary: Définition")

            st.write("\n\n")

            st.write('''
                Un **arbre de décision** est un modèle de Machine Learning utilisé à la fois en **régression** et en **classification**. \n
                Le modèle cherche à séparer les individus en groupes les plus “homogènes” possible par rapport à la variable cible. Plus ils sont homogènes, plus le modèle est performant. \n
                Le **DecisionTreeClassifier** est un algorithme de machine learning supervisé qui classe les données en 
                suivant une structure en **arbre**. Chaque **nœud** représente une question basée sur une caractéristique, 
                et chaque branche correspond à une réponse possible, jusqu'à atteindre une **feuille** qui donne la 
                classe finale.
            ''')

            st.write("---")

            st.write("### :material/manufacturing: Fonctionnement du modèle")

            st.write("\n\n")

            st.write('''
                * L’algorithme choisit la **meilleure question (feature)** pour séparer les données\n
                * Il divise les données en **branches** en fonction des réponses\n
                * Ce processus continue jusqu'à atteindre une **feuille** (classe finale)\n      
            ''')

            st.write("---")

            st.write("### :material/add_circle: Avantages du modèle")

            st.write("\n\n")

            st.write('''
                * Facile à comprendre et à visualiser\n
                * Fonctionne bien avec peu de prétraitement\n
                * Gère les données quantitatives et catégorielles\n
            ''')

            st.write("---")

            st.write("### :material/do_not_disturb_on: Inconvénient du modèle")

            st.write("\n\n")

            st.write('''
                * Peut souffrir d’**overfitting** si l’arbre est trop profond\n
            ''')

            st.write("---")

            st.write("### :material/labs: Résultats du modèle et analyses")

            st.write("\n\n")

            col1, col2 = st.columns(2, gap="large")

            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_Decision_Tree_Sans_Hyperparametre.png",caption="Decision Tree sans hyper paramètres")
                
                st.write("\n")
                st.write("\n")

                with st.expander("Overfitting détecté", icon=":material/format_text_overflow:") :
                    st.write('''
                        Il y a une très grosse différence entre le score du jeu d’entrainement et le score du jeu de test. Notre modèle n’apprend pas à généraliser. Le recall est faible, et l’accuracy n’est que de 55% seulement. C’est une mauvaise performance globale.\n
                        Notre modèle est adapté uniquement aux données d’entrainement mais ne fonctionne pas bien sur les nouvelles données.
                    ''')

            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_Decision_Tree_Avec_Hyperparametre.png",caption="Decision Tree avec hyper paramètres (max_depth=8)")
                
                st.write("\n")

                with st.expander("Meilleure performance globale", icon=":material/settings:") :
                    st.write('''
                        L’ajout d’un hyper paramètre, rend notre modèle plus performant en réduisant l’overfitting. La différence de score entre notre jeu de test et de notre jeu d’entrainement est considérablement réduite. Le résultat, qui s’élève à 0,62 (jeu d’entrainement et jeu de test), n’est pas aussi bon que le premier modèle car sur le premier nous avons un résultat de 0,99 (jeu d’entrainement) et 0,65 (jeu de test).\n
                        Le modèle peut malgré tout être considéré comme plus performant que la première version car il se trompe moins dans la prédiction.
                    ''')
                with st.expander("Faible recall, donc déséquilibre de classe", icon=":material/trending_down:") :
                    st.write('''
                        Toutefois, le recall (0,452) reste faible. Dans le détail, en réduisant notre max_depth à 10 nous avons réduit l’overfitting mais nous avons déséquilibré nos classes.\n
                        La détection des vrais positifs :\n
                        * s’améliore nettement lorsque grav = 1 (recall = 0,68 à recall = 0,87)\n
                        * s’améliore très légèrement lorsque grav = 3 (recall = 0,41 à recall = 0,47)\n
                        * diminue très légèrement lorsque grav = 4 (recall = 0,52 à recall = 0,47)\n
                        * devient nul lorsque grav = 2 (recall = 0,16 à recall = 0,00)
                    ''')

        with tab4 :

            st.write("\n\n\n\n\n\n\n\n\n\n")

            st.write("### :material/dictionary: Définition")

            st.write("\n\n")

            st.write('''
                Le **CatBoostClassifier** est un algorithme de machine learning supervisé qui appartient à la famille des 
                **boosting de gradient**. \n
                Il est conçu pour gérer efficacement les **données catégorielles** sans 
                transformation préalable, contrairement à d'autres algorithmes qui nécessitent de faire de l’encodage.
            ''')

            st.write("---")

            st.write("### :material/manufacturing: Fonctionnement du modèle")

            st.write("\n\n")

            st.write('''
                *  Construit des arbres de décision successifs, où chaque nouvel arbre corrige les erreurs du précédent\n
                *  Il remplace les catégories par des **valeurs numériques basées sur la cible**\n
                *  Il applique cette transformation **de manière progressive** pour éviter les fuites de données (data leakage)\n
                *  Il utilise une technique appelée **Ordre des permutations** pour éviter les biais d'entraînement\n
                *  Il applique un **boosting symétrique** qui construit **tous les arbres en parallèle**, ce qui accélère l'entraînement et améliore la stabilité\n      
            ''')

            st.write("---")

            st.write("### :material/add_circle: Avantages du modèle")

            st.write("\n\n")

            st.write('''
                * **Performant** sur les petits et grands ensembles de données\n
                * **Gère automatiquement** les variables catégoriques (pas besoin de One-Hot Encoding)\n
                * **Moins sensible au surapprentissage** (overfitting)\n
                * **Rapide et efficace**, même avec des données manquantes\n
            ''')

            st.write("---")

            st.write("### :material/do_not_disturb_on: Inconvénient du modèle")

            st.write("\n\n")

            st.write('''
                *  Temps d’entraînement plus long\n
                *  Moins flexible sur le réglage des hyperparamètres\n
                *  Besoin de plus de mémoire\n
            ''')

            st.write("---")

            st.write("### :material/labs: Résultats du modèle et analyses")

            st.write("\n\n")

            col1, col2 = st.columns(2, gap="large")

            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_CatBoost_Avec_Hyperparametre_Iteration_500.png", caption="CatBoost avec hyper paramètres: iterations=500")
                
                st.write("\n")

                with st.expander("Pas de présence d'un gros overfitting", icon=":material/fit_screen:") :
                    st.write('''
                        Le modèle généralise bien, avec une très faible différence entre l’entraînement et le test, ce qui indique qu’il n’y a pas d’overfitting important.
                    ''')
                with st.expander("Amélioration de la détection des cas positifs", icon=":material/trending_up:") :
                    st.write('''
                    Le modèle généralise bien, avec une très faible différence entre l’entraînement et le test, ce qui indique qu’il n’y a pas d’overfitting important.\n
                    Recall = 0.497 : le modèle détecte mieux les cas positifs comparé à d’autres modèles précédents. Il capture un nombre raisonnable de vrais positifs.\n
                    Une bonne précision et un bon recall pour la classe 1, ce qui indique que le modèle fait un bon travail sur la classe majoritaire.\n
                    La classe 2 reste apparemment mal captée par le modèle.\n
                    Les classes 3 et 4 sont bien mieux traitées même si elles ne sont pas parfaites.
                    ''')

            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_CatBoost_Avec_Hyperparametre_Iteration_500_Learning_Rate_01.png", caption="CatBoost avec hyper paramètres: iterations=500 et learning_rate=0.01")
                
                st.write("\n")

                with st.expander("Timide réduction de l'overfitting par rapport au modèle de gauche", icon=":material/trending_up:") :
                    st.write('''
                        Très légère amélioration par rapport au modèle de gauche qui montre une bonne généralisation et une petite diminution du surapprentissage par rapport à l’itération sans learning_rate = 0,01.
                    ''')
                with st.expander("Recall équivalent au modèle de gauche", icon=":material/equal:") :
                    st.write('''
                    Le recall est pratiquement identique à celui de l’itération de gauche, ce qui montre que les performances générales n’ont pas significativement changées avec le nouveau learning_rate.\n
                    La classe 2 continue à avoir un recall très faible. Le modèle a du mal à capturer cette classe.\n
                    Pour la classe 3 et la classe 4, le recall reste modéré.
                    ''')

            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col3, col4 = st.columns(2, gap="large")

            with col3 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_CatBoost_Avec_Hyperparametre_Iteration_1000_Learning_Rate_01.png", caption="CatBoost avec hyper paramètres: iterations=1000 et learning_rate=0.1")
                
                st.write("\n")

                with st.expander("Un overfitting qui continue de décroitre comparé aux itérations ci-dessus", icon=":material/trending_down:") :
                    st.write('''
                        Ces résultats montrent que notre modèle est très stable et continue à bien généraliser, avec une faible différence entre l’entraînement et le test, ce qui signifie moins de surapprentissage.
                    ''')
                with st.expander("Un recall qui repart légèrement à la baisse par rapport aux modèles ci-dessus", icon=":material/trending_down:") :
                    st.write('''
                        Recall = 0,482 : Bien que légèrement plus faible que ceux obtenus avec des hyperparamètres plus élevés pour le taux d'apprentissage, ces résultats restent raisonnables et montrent que le modèle arrive à identifier une proportion significative de vrais positifs.\n
                        Notre classe 2 reste très mal prédite.\n
                        Les classes 3 et 4 restent encore modérées, cela suggère que notre modèle a encore des difficultés à prédire correctement ces classes, notamment pour la classe 3.
                    ''')

            with col4 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Resultat_Machine_Learning/Resultat_CatBoost_Avec_Hyperparametre_Iteration_1000_Learning_Rate_001.png", caption="CatBoost avec hyper paramètres: iterations=1000, learning_rate=0.01 et depth=5")
                
                st.write("\n")

                with st.expander("Le modèle avec le plus faible écart entre le jeu d'entrainement et le jeu de test", icon=":material/rewarded_ads:") :
                    st.write('''
                        Les scores montrent une très faible différence entre l’entraînement et le test, ce qui est un bon signe de généralisation sans surapprentissage.
                    ''')
                with st.expander("Un recall qui diminue encore par rapport aux 3 autres modèles", icon=":material/trending_down:") :
                    st.write('''
                    Recall = 0,479 : Les performances restent stables et relativement raisonnables, bien que légèrement inférieures à celles d'autres configurations. Le modèle continue à bien détecter une proportion de vrais positifs dans les classes.\n
                    Toutefois, les problèmes persistent pour la classe 2 et les classes 3 et 4.\n
                    Notre modèle continue de donner des bons résultats avec très peu de surapprentissage, mais il reste des problèmes dans la prédiction des classes minoritaires.
                    ''')

        with tab5 :

            st.write('')

            # Variables prédictives (features) et variable cible
            X = df_accidents[['mois','annee', 'jour_semaine','heure','agg','col','dep','catr','situ','catu','catv','obsm','choc','age','place','manv']]
            y = df_accidents['grav']

            # Séparation en données d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Chemins des modèles pré-entrainés

            # Random Forest
            model_RF_0 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/RF_5.joblib"
            model_RF_5 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/RF_5.joblib"
            model_RF_6 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/RF_6.joblib"
            model_RF_7 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/RF_7.joblib"
            model_RF_8 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/RF_8.joblib"

            # Decision Tree
            model_DT_0 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/DT_5.joblib"
            model_DT_5 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/DT_5.joblib"
            model_DT_6 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/DT_6.joblib"
            model_DT_7 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/DT_7.joblib"
            model_DT_8 = r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/DT_8.joblib"

            # cat Boost
            model_CBC_I500=r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/CBC_I500.joblib"
            model_CBC_I500_LR_01=r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/CBC_I500_LR0.1.joblib"
            model_CBC_I1000_LR_001=r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/CBC_I1000_LR0.01.joblib"
            model_CBC_I1000_LR_001_D5=r"https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Joblibs/CBC_I1000_LR0.01_D5.joblib"

            # Fonction pour charger le modèle selon le choix de l'utilisateur
            @st.cache_data 
            def modele_machine_learning(model, profondeur):
                try :
                    if model == "Random Forest Classifier":
                        if profondeur == 0:
                            return joblib.load(BytesIO(requests.get(model_RF_0).content))
                        elif profondeur == 5:
                            return joblib.load(BytesIO(requests.get(model_RF_5).content))
                        elif profondeur == 6:
                            return joblib.load(BytesIO(requests.get(model_RF_6).content))
                        elif profondeur == 7:
                            return joblib.load(BytesIO(requests.get(model_RF_7).content))
                        elif profondeur == 8:
                            return joblib.load(BytesIO(requests.get(model_RF_8).content))
                    elif model == "Decision Tree Classifier":
                        if profondeur == 0:
                            return joblib.load(BytesIO(requests.get(model_DT_0).content))
                        elif profondeur == 5:
                            return joblib.load(BytesIO(requests.get(model_DT_5).content))
                        elif profondeur == 6:
                            return joblib.load(BytesIO(requests.get(model_DT_6).content))
                        elif profondeur == 7:
                            return joblib.load(BytesIO(requests.get(model_DT_7).content))
                        elif profondeur == 8:
                            return joblib.load(BytesIO(requests.get(model_DT_8).content))
                except KeyError:
                    st.write('')
                    return None
                except Exception as e:
                    st.write("Erreur lors du chargement du modèle :", e)
                    return None
            
            # # Fonction pour charger le modèle selon le choix de l'utilisateur
            # @st.cache_data 
            # def modele_ml_cb (model) :
            #     try :
            #         if model == "CatBoost_Iteration_500":
            #             return joblib.load(BytesIO(requests.get(model_CBC_I500).content))
            #         elif model == "CatBoost_Iteration_500_Learning_rate_0.1":
            #             return joblib.load(BytesIO(requests.get(model_CBC_I500_LR_01).content))
            #         elif model == "CatBoost_Iteration_1000_Learning_rate_0.01":
            #             return joblib.load(BytesIO(requests.get(model_CBC_I1000_LR_001).content))
            #         elif model == "CatBoost_Iteration_1000_Learning_rate_0.1_Depth_5":
            #             return joblib.load(BytesIO(requests.get(model_CBC_I1000_LR_001_D5).content))
            #     except KeyError:
            #         st.write('')
            #         return None
            #     except Exception as e:
            #         st.write("Erreur lors du chargement du modèle :", e)
            #         return None
                
            # Fonction pour calculer les scores sur les données d'entraînement
            def train_scores(model, metrique):
                y_pred = model.predict(X_train)  
                if metrique == "Accuracy":
                    return accuracy_score(y_train, y_pred)
                elif metrique == "Recall":
                    return recall_score(y_train, y_pred, average="macro")
                elif metrique == "F1-score":
                    return f1_score(y_train, y_pred, average='macro')
                elif metrique == "Classification Report":
                    return classification_report(y_train, y_pred)
                else:
                    return "Métrique invalide"

            # Fonction pour calculer les scores sur les données de test
            def test_scores(model, metrique):
                y_pred = model.predict(X_test)
                if metrique == "Accuracy":
                    return accuracy_score(y_test, y_pred)
                elif metrique == "Recall":
                    return recall_score(y_test, y_pred, average="macro")
                elif metrique == "F1-score":
                    return f1_score(y_test, y_pred, average='macro')
                elif metrique == "Classification Report":
                    return classification_report(y_test, y_pred)
                else:
                    return "Métrique invalide"

            # Interface Streamlit
            left_co, right_co = st.columns(2)
            with left_co:
                # Choisir le modèle
                choix_modele = ["Random Forest Classifier", "Decision Tree Classifier"]
                modele_choisi = st.selectbox("Choisissez un modèle de Machine Learning :", choix_modele, index=None, placeholder="cliquez ici ...")
                st.write("Vous avez choisi le modèle : ", modele_choisi)
                st.write('')
                # Choisir la profondeur
                profondeur_choisi = st.slider("Choisissez une profondeur (max_depth) pour votre modèle :", 5, 8)
                st.write("Vous avez choisi une profondeur de :", profondeur_choisi)

                # Charger le modèle sélectionné
                model = modele_machine_learning(modele_choisi, profondeur_choisi)

                # Choisir la métrique à afficher
                display = st.radio("Quelle métrique souhaitez-vous afficher ?", ("Accuracy", "Recall", "F1-score", "Classification Report"))

                # Si un modèle est chargé
                if model:  
                    if display == "Accuracy":
                        st.write("Score sur le jeu d'entrainement :", train_scores(model, display))
                        st.write("Score sur le jeu de test :", test_scores(model, display))
                    elif display == "Recall":
                        #st.write("Score d'entraînement:", train_scores(model, display))
                        st.write("Recall :", test_scores(model, display))
                    elif display == "F1-score":
                        #st.write("Score d'entraînement:", train_scores(model, display))
                        st.write("F1-score :", test_scores(model, display))
                    elif display == "Classification Report":
                        #st.text(train_scores(model, display))
                        st.text(test_scores(model, display))

                # st.write("\n\n\n\n\n")
                # st.write("---")
                # st.write("\n\n\n\n\n")

                # # Choisir le modèle
                # choix_modele_ml_cb = ["CatBoost_Iteration_500", "CatBoost_Iteration_500_Learning_rate_0.1", "CatBoost_Iteration_1000_Learning_rate_0.01", "CatBoost_Iteration_1000_Learning_rate_0.1_Depth_5"]
                # modele_ml_cb_choisi = st.selectbox("Choisissez un modèle de Machine Learning :", choix_modele_ml_cb, index=None, placeholder="cliquez ici ...")
                # st.write("Vous avez choisi le modèle : ", modele_ml_cb_choisi)
                # st.write('')

                # # Charger le modèle sélectionné
                # model_cb = modele_ml_cb(modele_ml_cb_choisi)

                # # Choisir la métrique à afficher
                # display_cb = st.radio("Quelle métrique souhaitez-vous afficher ? ", ("Accuracy", "Recall", "F1-score", "Classification Report"))

                # # Si un modèle est chargé
                # if model_cb:
                #     if display_cb == "Accuracy":
                #         st.write("Score sur le jeu d'entrainement :", train_scores(model_cb, display_cb))
                #         st.write("Score sur le jeu de test :", test_scores(model_cb, display_cb))
                #     elif display_cb == "Recall":
                #         #st.write("Score d'entraînement:", train_scores(model, display))
                #         st.write("Recall :", test_scores(model_cb, display_cb))
                #     elif display_cb == "F1-score":
                #         #st.write("Score d'entraînement:", train_scores(model, display))
                #         st.write("F1-score :", test_scores(model_cb, display_cb))
                #     elif display_cb == "Classification Report":
                #         #st.text(train_scores(model, display))
                #         st.text(test_scores(model_cb, display_cb))

    if page == pages[4] :

        st.write("\n\n\n\n\n")

        st.write("### :material/precision_manufacturing: Quel modèle est le plus performant ?")

        st.write("\n\n")

        with st.expander("**1.** Modèle CatBoostClassifier avec hyper paramètres : itérations = 500 et learning_rate = 0.1", icon=":material/lightbulb_circle:") :
            st.write('''
                    Ce modèle semble le plus performant en termes de score de test et recall. Bien qu’il représente un légèrement meilleur score d’entrainement (0,673), il ne souffre pas d’overfitting important, ce qui est bon signe de généralisation.
                ''')

        with st.expander("**2.** Modèle CatBoostClassifier avec hyper paramètres : itérations = 1000 et learning_rate =0,01", icon=":material/lightbulb:") :
            st.write('''
                    Ce modèle est également assez bon en termes de généralisation, mais il ne surpasse pas le premier en terme de recall ou d’accuracy, bien que le recall (0,479) reste assez proche de celui du deuxième modèle.
                ''')  

        st.write("---")

        st.write("### :material/all_inclusive: Meilleur choix ?")

        st.write("\n\n")

        st.write('''
            La recherche d’un modèle de Machine Learning performant est un juste équilibre à trouver entre un overfitting le plus faible possible, et un recall le plus élevé possible. En jouant avec les hyperparamètres nous avons su réduire l’overfitting, au profit d’un recall plus faible que lors de la première itération sur nos différents modèles.\n
            Malheureusement, nous n’avons pas su trouver les bons paramètres afin d’augmenter la valeur du recall, ce qui fait du modèle choisi un modèle de prédiction peu performant.
        ''')  
        
        st.write("---")

        st.write("### :material/work_history: Feature Importance")

        st.write("\n\n")

        with st.popover("Dictionnaire des variables", icon=":material/dictionary:") :
                st.dataframe(df_description_variables, hide_index=True)

        with st.expander("**1.** DécisionTreeClassifier/RandomForestClassifier", icon=":material/forest:") :
            col1, col2 = st.columns(2, gap="small")
            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/D%C3%A9cisionTreeClassifier.png", width = 500, caption="DecisionTreeClassifier - Max_Depth=10 ")
            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/RandomForestClassifier.png", width = 500, caption="RandomForestClassifier - Max_Depth=8")

        with st.expander("**2.** CatBoostClassier", icon=":material/pets:") :
            col1, col2 = st.columns(2, gap="small")
            with col1 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/CatBoostClassifier%20500.png", width = 500, caption="CatBoostClassifier - Learning_rate=0.1 et Iterations=500")
            with col2 :
                st.image("https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/CatBoostClassifier%201000.png", width = 500, caption="CatBoostClassifier - Learning_rate=0.01 et Iterations=1000")

        st.write("---")

        st.write("### :material/emoji_objects: Pistes pour améliorer les résultats de notre Machine Learning")

        st.write("\n\n")

        with st.expander("**1.** Il existe des variables dans le dataset qui sont plus corrélées à la variable cible que les variables sélectionnées", icon=":material/variable_insert:") :
            st.write('''
                    La lecture du recall nous laisse à penser que si le modèle est performant sur certaines classes c’est que nous n’avons pas choisi des variables saillantes permettant de mieux catégoriser la gravité des accidents.
                ''')

        with st.expander("**2.** Les hyperparamètres des modèles", icon=":material/online_prediction:") :
            st.write('''
                    Nous avons manqué de temps pour étudier l’ensemble des hyperparamètres des modèles pour essayer d’améliorer notre modèle. La compréhension des hyperparamètres est une étape importante pour jouer sur les résultats.
                ''')  

        with st.expander("**3.** Modifier la méthodologie de travail", icon=":material/tactic:") :
            st.write('''
                    Après réflexion, nous pensons qu’il aurait fallu avoir une approche différente sur notre modélisation. Nous aurions dû lancer nos modèles sans hyperparamètres et jouer avec RandomSearch et GridSearch pour approcher, puis trouver, les meilleurs paramètres à appliquer sur nos modèles
                ''')
            
        st.write("---")

        st.write("### :material/search: Conclusion")

        st.write("\n\n")      

        st.write('''  
            La gravité des accidents de la route en France est un sujet de préoccupation national, tant pour les autorités, les citoyens et les associations de prévention routière.\n
            La gravité de ceux-ci est dû à plusieurs facteurs :\n
            * La vitesse excessive\n
            * L’alcool\n
            * La drogue\n
            * La fatigue\n
            * Le non-respect du code la route\n
            Dans les données communiquées par l’Observatoire national interministériel de la sécurité routière, certaines données ne sont pas communiquées comme l’alcool, la drogue, et la vitesse au moment de l’accident.\n
            Nous pensons que ces dimensions auraient pu apporter des résultats encore plus probants puisqu’ils ont lié fortement à la gravité des accidents d’après les statistiques de la sécurité routière.\n
            L’accidentologie est un enjeu de santé publique majeur, et chaque année le gouvernement tente d'enrayer la propagation d'accident en cherchant des solutions.
            Il est important de continuer à investir sur la compréhension des phénomènes des accidents et des leurs liens avec la gravité pour adapter l’urbanisme, et sensibiliser la population afin d’enrayer ce fléau qui a encore causé 3.431 décès sur les routes en France Métropolitaine ou d’Outre-Mer.
        ''')    
except Exception as e :
    st.error(f"Erreur critique lors du démarrage : {e}")
    st.text(traceback.format_exc())