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

@st.cache_data
def charger_datasets():

    df_caracteristiques=pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\caracteristiques.csv", encoding = "ISO-8859-1",index_col=0)
    df_lieux = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\lieux.csv", encoding = "ISO-8859-1",index_col=0)
    df_usagers = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\usagers.csv", encoding = "ISO-8859-1",index_col=0)
    df_vehicules = pd.read_csv(r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\vehicules.csv", encoding = "ISO-8859-1",index_col=0)
    df_description_variables = pd.read_excel(BytesIO(requests.get("https://github.com/Kaalinodi57/accidents-routes-cda/raw/refs/heads/main/Projets%20accidents%20-%20Description%20des%20variables.xlsx").content))
    return df_caracteristiques, df_lieux, df_usagers, df_vehicules, df_description_variables

st.set_page_config(layout="wide")
st.title("Prédiction de la gravité des accidents :collision:")

df_caracteristiques, df_lieux, df_usagers, df_vehicules, df_description_variables = charger_datasets()

#Chargement des datasets
#caracteristiques = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\caracteristiques.csv"
#lieux = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\lieux.csv"
#usagers = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\usagers.csv"
#vehicules = r"C:\Users\adilb\OneDrive\Documents\Projets\Formation Data Analyst\Projet\Cohorte août 24 - Projet accidents\0 - Jeux de données\vehicules.csv"
#description_variables = r"C:\Users\adilb\Downloads\Projets accidents - Description des variables.xlsx"
#df_caracteristiques = pd.read_csv(caracteristiques, encoding = "ISO-8859-1", index_col=0)
#df_lieux = pd.read_csv(lieux, encoding = "ISO-8859-1",index_col=0)
#df_usagers = pd.read_csv(usagers, encoding = "ISO-8859-1",index_col=0)
#df_vehicules = pd.read_csv(vehicules, encoding = "ISO-8859-1",index_col=0)
#df_description_variables = pd.read_excel(description_variables)

#Titre du streamlit
#st.set_page_config(layout="wide")
#st.title("Prédiction de la gravité des accidents :collision:")

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

    st.write("### :material/fit_screen: Réduction de la profondeur des données")

    st.write("\n\n")

    with st.expander("Nous ne conservons que les accidents entre 2005 et 2017, soit 12 ans d'historique", icon=":material/delete_history:") :
        st.write('''
            Lors du lancement de notre projet Fil Rouge, nous avons pris la décision de prendre la base de 
            données contenant la totalité des informations de 2005 à 2021 pour chacun des 4 fichiers 
            (Caractéristiques – Lieux – Véhicules – Usagers) au format CSV. \n
            Nous avons fait le choix de conserver les données sur les 12 premières années (2005 à 2017) pour 
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
        st.write("### :material/road: :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par catégorie de route")

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
        st.write("### :material/road: :material/pin_drop: :material/car_crash: :material/personal_injury: Accidents par localisation sur la route")

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
        with st.expander("Les fragilités physiques et comportements routiers spécifiques aux jeunes personnes ont une implication dans la gravité des accidents auxquels ils sont soumis", icon=":material/personal_injury:") :
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
                Nous effectuons la fusion en trois temps puisque les clés de jointure ne sont pas identiques).
            ''')

    with st.expander("**D.** Supprimer les colonnes en doublon (annee_[…], num_veh_[…]) dans le df_accidents", icon=":material/delete:") :
        with st.popover("Supprimer les colonnes annee_[…] en doublon dans le df_accidents") :
            st.write('''
                Nous ne conserverons qu’une seule « année » pour notre dataset final. Nous supprimons les colonnes "annee_vehicules", "annee_usagers", "annee_lieux").
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
        st.markdown("<h6 style='text-align: center; color: grey;'>Pour chaque variable, nous avons effectué le travail suivant : </h6>", unsafe_allow_html=True)
        st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Travail%20effectu%C3%A9%20sur%20chaque%20variable.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
        st.write("\n\n\n\n\n")



    with st.expander("**H.** Variables conservées", icon=":material/dataset:") :
        st.write('''
            Pour donner suite à notre analyse lors de la visualisation, nous décidons de supprimer les colonnes non pertinentes pour les modèles dans le df_accidents. 
        ''')
        st.markdown("<h6 style='text-align: center; color: grey;'>Après la suppression notre dataset df_accidents est composé des colonnes suivantes : </h6>", unsafe_allow_html=True)
        st.markdown("<img src='https://raw.githubusercontent.com/Kaalinodi57/accidents-routes-cda/refs/heads/main/Images/Variables%20conserv%C3%A9s.png' width='500' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
        st.write("\n\n\n\n\n")

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