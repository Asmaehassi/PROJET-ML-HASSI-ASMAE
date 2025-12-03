#  **Projet Machine Learning : Analyse de la Santé Mentale**
# **Asmae Hassi**  
 

---

## 1. Introduction


Cette étude vise à analyser un ensemble de données portant sur des patients suivis en santé mentale, dans le but de prédire l’issue de leur traitement, définie comme une amélioration ou une non-amélioration. L’objectif est de transformer des données brutes en informations exploitables et de développer un modèle prédictif fiable permettant d’anticiper l’évolution clinique des patients.

Le dataset utilisé, intitulé “Mental Health Diagnosis and Treatment Monitoring” et issu de la plateforme Kaggle, regroupe des variables cliniques, psychologiques et comportementales permettant de décrire l’état de chaque patient. Ce jeu de données se prête particulièrement bien à un projet de Machine Learning supervisé, car il contient une variable cible clairement définie (Outcome), exprimée sous forme binaire, ainsi qu’un ensemble de variables explicatives diversifiées, cohérentes et suffisamment structurées pour permettre un prétraitement efficace.

L’intérêt de ce dataset réside dans sa capacité à représenter des problématiques réelles en santé mentale, notamment à travers des mesures telles que la sévérité des symptômes, le niveau de stress, la qualité du sommeil, la progression du traitement ou encore l’adhésion thérapeutique. L’objectif principal de son exploitation est de modéliser et prédire l’évolution d’un patient en fonction de ces indicateurs, afin d’identifier les facteurs influençant le succès du traitement et d’appuyer la prise de décision clinique.

 ### Problématique

 Peut-on prédire l’évolution d’un patient en santé mentale (amélioration ou non) à partir de données psychologiques, physiologiques et comportementales ?
 

## 2. Description du Dataset

Le dataset *Mental Health Diagnosis and Treatment Monitoring* contient **500 lignes et 17 colonnes**.  
Les données décrivent des patients, leurs symptômes, leur traitement et l’issue observée. :contentReference[oaicite:0]{index=0}

## 3. Structure des données  


| Colonne | Type | Description |
|---------|------|-------------|
| patient_id | entier | Identifiant unique patient |
| age | entier | Âge du patient |
| gender | chaîne | Genre (Male / Female) |
| diagnosis | chaîne | Diagnostic clinique déclaré |
| symptom_severity_1_10 | entier | Sévérité des symptômes (1–10) |
| mood_score_1_10 | entier | État d’humeur (1–10) |
| sleep_quality_1_10 | entier | Qualité du sommeil (1–10) |
| physical_activity_hrs_week | entier | Activité physique hebdomadaire |
| medication | chaîne | Médication utilisée |
| therapy_type | chaîne | Type de thérapie suivie |
| treatment_start_date | timestamp | Date de début de traitement |
| treatment_duration_weeks | entier | Durée du traitement en semaines |
| stress_level_1_10 | entier | Niveau de stress (1–10) |
| outcome | chaîne | Issue du traitement (target) |
| treatment_progress_1_10 | entier | Progression du traitement |
| ai_detected_emotional_state | chaîne | État émotionnel détecté par IA |
| adherence_to_treatment | entier | Respect du traitement (%) |

Ce dataset a été conçu pour étudier l’évolution des patients tout au long du traitement, en associant des variables cliniques et comportementales au résultat final (“outcome”). :contentReference[oaicite:1]{index=1}

 ## Objectifs du projet

- Identifier les variables les plus influentes dans l’évolution d’un patient.

- Construire et comparer plusieurs modèles prédictifs (Logistic Regression, Random Forest, XGBoost).

- Optimiser les performances du modèle le plus performant via GridSearchCV.

- Évaluer les performances via les métriques standards : accuracy, F1-score, matrice de confusion.

---
## Méthodologie

##  4. Prétraitement des données

Les étapes principales de nettoyage et de prétraitement ont été :

- **Suppression des doublons**
- **Gestion des valeurs manquantes**
- **Encodage des variables catégorielles**, notamment `gender` et `outcome`
- **Standardisation des variables numériques** pour faciliter l’apprentissage des modèles
- **Séparation en ensembles d’entraînement (80%) et de test (20%)**
- #### Encodage

- Outcome encodé en binaire via LabelEncoder.

- Gender encodé en valeurs numériques (0/1).

### Normalisation

- Utilisation de StandardScaler pour homogénéiser l’échelle des variables numériques.

#### Justification : 
- nécessaire pour les modèles sensibles aux valeurs (Logistic Regression, SVM, XGBoost).
### Colonnes supprimées & justification

#### Les colonnes suivantes ont été supprimées car :

- elles n’étaient pas exploitables,

- elles ne contribuaient pas à la performance prédictive

## Colonnes supprimées :

- Patient IDv
- Diagnosis
- Medication
- Therapy Type
- Treatment Start Date
- AI-Detected Emotional Stat


---
##  5.Implémentation Technique et Description du Code
Dans cette section, nous présentons l’ensemble des blocs de code développés pour mener à bien le projet de Machine Learning. Chaque portion de code est accompagnée d’une brève description permettant de comprendre son rôle dans la chaîne de traitement : préparation des données, exploration, modélisation, optimisation et évaluation.
L’objectif est d’exposer clairement la démarche méthodologique suivie et de justifier les choix techniques réalisés, conformément au cahier des charges.

### 5.1 Importation des librairies

 ```import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
```
Ce bloc importe toutes les librairies nécessaires pour :


- Manipuler les données (Pandas, Numpy)

- Faire des graphiques (Matplotlib, Seaborn)

- Faire le prétraitement (encodage, scaling)

- Tester plusieurs modèles de Machine Learning

- Mesurer la performance (accuracy, classification report)
  
### 5.2 Chargement du dataset

  
```url = "https://raw.githubusercontent.com/Asmaehassi/PROJET-ML-HASSI-ASMAE/main/mental_health_diagnosis_treatment_.csv"
dataset = pd.read_csv(url)
dataset.head()
```

Le dataset est importé directement depuis GitHub pour faciliter l’exécution en ligne.
### 5.3 Nettoyage (Suppression des colonnes inutiles)
```
listDrop = ["Patient ID", "Diagnosis", "Medication", "Therapy Type", 
            "Treatment Start Date", "AI-Detected Emotional State"]

for col in listDrop:
    dataset = dataset.drop(col, axis="columns")

```

Ces colonnes contiennent des informations non pertinentes ou difficilement exploitables par le modèle.
### 5.4 Séparation des variables explicatives et de la cible

```X = dataset[["Age","Gender","Symptom Severity (1-10)","Mood Score (1-10)",
             "Sleep Quality (1-10)","Physical Activity (hrs/week)",
             "Treatment Duration (weeks)", "Stress Level (1-10)",
             "Treatment Progress (1-10)","Adherence to Treatment (%)"]]

y = dataset["Outcome"]
```


X contient les variables utilisées pour prédire

y est la variable cible ("Outcome")

###  Encodage des variables catégorielles

```le = LabelEncoder()
y = le.fit_transform(dataset["Outcome"])
X["Gender"] = LabelEncoder().fit_transform(X["Gender"])
```

Les algorithmes nécessitent des valeurs numériques → on transforme le texte en chiffres.

###  Séparation Train / Test

```X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)
```

Le dataset est divisé en :

80% entraînement

20% test

###  Standardisation des variables
```
scaler = StandardScaler()

cols_to_scale = [0, 2, 3, 4, 5, 6, 7, 8, 9]
X_train.iloc[:, cols_to_scale] = scaler.fit_transform(X_train.iloc[:, cols_to_scale])
X_test.iloc[:, cols_to_scale] = scaler.transform(X_test.iloc[:, cols_to_scale])
```

Les variables numériques sont normalisées pour améliorer les performances des modèles linéaires.

## 6. Analyse Exploratoire (EDA)
### 6.1 Histogrammes
   
```dataset.hist(figsize=(12,8))
plt.show()
```

Visualise la répartition des valeurs (âge, stress, humeur, sommeil…).

### 6.2 Boxplots

```Plt.figure(figsize=(10,6))
sns.boxplot(data=dataset)
plt.xticks(rotation=90)
plt.show()
```


Permet d’identifier les outliers et les variations dans les variables.

### 6.3 Heatmap (corrélations)

```plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
plt.show()
```

Montre les associations entre variables.
On observe par exemple une forte corrélation entre :

Treatment Progress et Outcome

Stress et qualité du sommeil

 ## 7. Modélisation (Machine Learning)
 
Trois modèles ont été testés conformément au cahier des charges.

### 7.1 Modèle 1 : XGBoost (modèle principal)

```ai = XGBClassifier()
ai.fit(X_train, y_train)

y_pred = ai.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

```

XGBoost est un modèle puissant et performant pour les données tabulaires.

### 7.2 Prédiction personnalisée

```custom_input = [[45, 1, 9, 3, 5, 8, 10, 6, 5, 60]]
prediction = ai.predict(custom_input)
print("Predicted class:", prediction[0])
```
### 7.3 Modèle 2 : Régression Logistique

```log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print(" Logistic Regression Accuracy :", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
```


Modèle simple mais très efficace pour une première baseline.

### 7.4 Modèle 3 : Random Forest

```rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy :", accuracy_score(y_test, y_pred_rf))
```

Modèle basé sur plusieurs arbres de décision → robuste et fiable.

 ### 7.5 Optimisation des hyperparamètres (GridSearchCV)

```from sklearn.model_selection import GridSearchCV

params = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "n_estimators": [50, 100, 200]
}

grid = GridSearchCV(
    estimator=XGBClassifier(),
    param_grid=params,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
```
Cette étape identifie automatiquement les meilleurs paramètres pour XGBoost.




#   Interprétation des histogrammes 

L’analyse des distributions à travers les histogrammes met en évidence plusieurs caractéristiques importantes du dataset, essentielles pour comprendre le comportement des variables et la qualité des données avant modélisation.

### **Age**

<img width="256" height="212" alt="image" src="https://github.com/user-attachments/assets/fb1dfec9-0372-4d4b-8d29-47e0e0e73588" />


La distribution de l’âge est relativement uniforme entre 20 et 60 ans, ce qui indique une population diversifiée en termes de profil démographique.
Aucune concentration extrême ne se dégage, ce qui limite les risques de biais liés à l’âge dans la modélisation.

### **Symptom Severity**

<img width="271" height="208" alt="image" src="https://github.com/user-attachments/assets/2c45fd7e-7de6-4ac6-9024-576b2f95f668" />


Les niveaux de sévérité des symptômes se situent majoritairement entre 6 et 9.
Cette concentration montre que la plupart des patients présentent une symptomatologie modérée à sévère, ce qui est cohérent avec un suivi en santé mentale.
### **Mood Score**

<img width="236" height="203" alt="image" src="https://github.com/user-attachments/assets/473a2a4b-5e80-456f-88b8-df532d4e927b" />

Le score d’humeur varie principalement entre 4 et 8, traduisant une dispersion modérée.
Cela suggère une population présentant des variations émotionnelles importantes, reflet fréquent dans un contexte de suivi psychologique.

### **Sleep Quality**

<img width="262" height="197" alt="image" src="https://github.com/user-attachments/assets/303c2158-be99-410b-8cb3-95356834845a" />

La qualité du sommeil est globalement homogène, oscillant entre 5 et 9.
Cela laisse supposer que les patients rapportent une qualité de sommeil acceptable à moyenne, sans extrêmes significatifs.
### **Physical Activity**

<img width="269" height="196" alt="image" src="https://github.com/user-attachments/assets/3ddf208d-cc9b-400f-9522-17b1f9a43e2f" />

La distribution est étendue, de 1 à 10 heures par semaine.
Cela traduit des différences notables dans les habitudes de vie, pouvant expliquer des variations dans la progression thérapeutique.

### **Treatment Duration**

<img width="244" height="198" alt="image" src="https://github.com/user-attachments/assets/607fd77d-0fe7-4a7f-8768-b2228c2d571f" />

La durée des traitements se situe généralement entre 8 et 16 semaines, ce qui montre une certaine cohérence des protocoles thérapeutiques.
Cette uniformité limite l’impact de la durée comme facteur de bruit.
### **Stress Level**

<img width="250" height="189" alt="image" src="https://github.com/user-attachments/assets/54f6b35e-d895-4c2c-ba6d-46ca159b377b" />

Le score de stress est fortement concentré entre 7 et 10, indiquant un niveau de stress globalement élevé chez la majorité des patients.
Cela reflète une caractéristique clinique souvent observée dans les suivis psychothérapeutiques.

### **Treatment Progress**

<img width="250" height="188" alt="image" src="https://github.com/user-attachments/assets/8509a511-7705-4beb-b5e0-5b7824ea43ea" />

Une répartition relativement équilibrée des niveaux de progression est observée.
Cette diversité des résultats thérapeutiques représente un atout pour la modélisation, permettant de distinguer efficacement les profils de patients.
### **Adherence to Treatment**

<img width="233" height="187" alt="image" src="https://github.com/user-attachments/assets/7e6a50ef-7308-4ee6-8353-c291732057f6" />

L’adhésion varie entre 60 % et 90 %, avec une concentration notable autour de 70–80 %.
Une adhésion relativement élevée constitue un indicateur positif de l’engagement thérapeutique des patients.

---

##  Interprétation des boxplots

Les boxplots offrent une vue synthétique de la dispersion des données et facilitent l’identification d’éventuelles anomalies.
<img width="1146" height="817" alt="image" src="https://github.com/user-attachments/assets/6cb58cab-7dfa-4d0f-b065-0d69a149a4a1" />


### **Age**

La distribution présente une dispersion modérée à forte, ce qui confirme l’hétérogénéité de l’échantillon en termes d’âge.
Aucun outlier extrême n’est détecté, ce qui suggère une collecte de données cohérente.

### **Symptom Severity**

La faible dispersion indique que les valeurs sont regroupées dans un intervalle serré, ce qui renforce l'idée d’une population présentant essentiellement des niveaux de symptômes similaires.
L’absence d’outliers traduit une mesure stable.

### **Mood Score**

Une variabilité modérée est observée sans outliers, suggérant des évaluations relativement cohérentes de l’état émotionnel des patients.

### **Sleep Quality**

La dispersion est limitée et aucun outlier n’est présent, indiquant une bonne cohérence interne dans les auto-évaluations de la qualité du sommeil.

### **Physical Activity**

La distribution affiche quelques valeurs plus élevées, mais celles-ci ne constituent pas des outliers au sens statistique strict.
La variabilité observée reflète les différences de mode de vie entre les patients.

### **Treatment Duration**

Une dispersion cohérente est observée entre 8 et 16 semaines, sans valeurs extrêmes.
Cela reflète une normalisation potentielle des durées des protocoles thérapeutiques.

### **Stress Level**

Les niveaux de stress sont élevés et relativement homogènes, sans outliers apparents.
Cette homogénéité pourrait indiquer que la variable est fortement liée à la condition psychologique globale des patients.

### **Treatment Progress**

La progression du traitement affiche une dispersion modérée, indiquant une diversité dans les trajectoires thérapeutiques, ce qui constitue un signal précieux pour la prédiction de l’Outcome.

### **Adherence**

La variabilité est importante, mais aucune valeur aberrante n’est détectée.
Les niveaux d’adhésion constituent probablement un facteur déterminant dans l’évolution clinique du patient.

---

## Synthèse EDA

L’EDA montre :

* Une **bonne qualité globale des données**
* Une **absence d’outliers majeurs**
* Une **cohérence clinique élevée**
* Une **variabilité suffisante** pour entraîner un modèle performant

Variables les plus informatives :

* **Stress Level**
* **Treatment Progress**
* **Symptom Severity**
* **Adherence to Treatment**

---

## Interprétation de la Matrice de Corrélation

La matrice de corrélation permet d’évaluer la force et la direction des relations linéaires entre les variables numériques. Les coefficients varient entre :

- **+1** : corrélation positive parfaite  
- **0** : absence de relation linéaire  
- **–1** : corrélation négative parfaite  

Dans ce dataset, les valeurs observées sont globalement faibles, ce qui traduit une faible interdépendance entre les variables.

<img width="1078" height="864" alt="image" src="https://github.com/user-attachments/assets/2b838c2e-35a2-4ee2-9246-190398bf37bb" />

---

##  Principales Observations

###   Corrélations faibles à négligeables 

La majorité des coefficients sont proches de zéro, indiquant l’absence de relation linéaire significative.

- **Symptom Severity** présente des corrélations très faibles avec toutes les autres variables (**|r| < 0.05**).  
  → Cela montre que, dans ce dataset, la sévérité des symptômes n’est pas linéairement liée à l’âge, au stress, au sommeil ou à l’activité physique.

- **Treatment Progress** montre une très légère corrélation positive avec :  
  - *Sleep Quality* (**r = 0.082**)  
  - *Mood Score* (**r = 0.052**)  
  → Tendances positives faibles, insuffisantes pour parler d'une relation significative.

---

###  Corrélations modérément négatives

Quelques relations négatives faibles apparaissent :

- **Age** & **Treatment Duration** → *r = -0.11*  
  → Les patients plus âgés ont tendance à suivre des traitements légèrement plus courts.

- **Mood Score** & **Treatment Duration** → *r = -0.079*  
  → Légère tendance à ce que les patients avec un meilleur score d’humeur suivent des traitements plus courts.

Ces valeurs restent très faibles et n’indiquent pas de relation forte.

---

###   Aucune corrélation notable

Certaines variables semblent quasiment indépendantes de toutes les autres :

- **Physical Activity** : |r| < 0.05 pour toutes les variables  
- **Stress Level** : corrélations très faibles  

Cela suggère que l’activité physique hebdomadaire et le niveau de stress ne sont pas linéairement associés aux autres indicateurs du dataset.

---

##   Implications pour l’Analyse

### Absence de Multicollinéarité  
Les faibles corrélations entre variables indiquent que celles-ci ne sont pas redondantes.  
 Les modèles prédictifs (régression, arbres, réseaux, etc.) ne seront pas pénalisés par une trop forte colinéarité.

###  Relations linéaires faibles  
Les variables expliquent très peu les variations des autres d’un point de vue linéaire.  
 Il est possible que les relations soient **non linéaires** ou influencées par des facteurs non inclus dans le dataset.

###  Impact sur les modèles prédictifs  
- Les modèles linéaires simples pourraient avoir des performances limitées.  
- Les modèles non linéaires (Random Forest, XGBoost) pourraient mieux capturer les interactions faibles ou complexes.  

###  Cibles possibles  
Si l’objectif est de prédire **Symptom Severity** ou **Treatment Progress**,  
 ces variables semblent peu corrélées aux autres, ce qui réduit la capacité explicative des simples modèles linéaires.

---
## Résultats & Discussion 

<img width="547" height="278" alt="image" src="https://github.com/user-attachments/assets/5b0dc9b5-e5b4-4d6f-9402-dd78b770f812" />


Le modèle XGBoost présente une performance globalement médiocre, avec une précision (accuracy) de seulement **0,31**, ce qui signifie qu'il ne parvient à prédire correctement la classe d'appartenance que pour environ **31%** des observations. Cette performance est extrêmement faible, se situant pratiquement au niveau d'une prédiction aléatoire (qui, pour un problème à trois classes équilibrées, aurait une accuracy attendue d'environ **33%**). Le rapport de classification par classe révèle que le modèle ne parvient à obtenir de scores de précision, de rappel (recall) et de F1-score que dans une fourchette de **0,25** à **0,36** pour chacune des trois classes, sans qu'aucune classe ne se distingue par une performance nettement supérieure. Cette uniformité dans la médiocrité des scores suggère que le modèle ne parvient pas à identifier de motifs distinctifs ou de relations solides dans les données pour discriminer efficacement les classes.

Ce résultat indique qu'une approche plus approfondie est nécessaire. Avant d'envisager un modèle plus complexe, il convient de vérifier la qualité du prétraitement des données, d'explorer l'importance des variables pour identifier les prédicteurs les plus informatifs, et de réfléchir à un éventuel feature engineering pour créer de nouvelles variables plus discriminantes. Une optimisation rigoureuse des hyperparamètres et une validation croisée stricte pourraient également améliorer les performances, mais la racine du problème semble résider dans la nature même des données ou dans la définition de la problématique prédictive.



**La matrice de confusion révèle une dispersion importante des prédictions :**

<img width="511" height="435" alt="image" src="https://github.com/user-attachments/assets/bacaf81c-494c-498c-8eef-7446fd11d649" />


- forte confusion entre les trois catégories

- absence d’un schéma clair de classification

Ces résultats soulignent l’hypothèse que les variables disponibles, bien qu’intéressantes, ne suffisent probablement pas à caractériser de manière robuste l’évolution clinique des patients. Il est également possible que le dataset contienne un niveau élevé de bruit, ou que les variables soient corrélées entre elles de manière complexe mais non suffisamment informative pour permettre une séparation nette des classes.

# Conclusion : Limites du modèle et pistes d'amélioration

L’étude réalisée sur le dataset *Mental Health Diagnosis and Treatment Monitoring* a permis de développer un modèle prédictif visant à anticiper l’issue du traitement psychothérapeutique des patients. Les résultats montrent que certaines variables, telles que le niveau de stress, la progression du traitement ou encore la sévérité des symptômes, jouent un rôle important dans la compréhension de l’évolution clinique.

Cependant, plusieurs limites doivent être prises en considération :

##  Limites du modèle  
- **Taille réduite du dataset :** Le nombre d’observations reste limité, ce qui réduit la capacité du modèle à généraliser et augmente le risque de surapprentissage.  
- **Corrélations faibles entre variables :** Les relations linéaires étant globalement faibles, les modèles simples (régression logistique, SVM linéaire) peuvent avoir du mal à capturer des patterns complexes.  
- **Variables auto-rapportées :** Plusieurs mesures (mood score, sleep quality, stress level) reposent sur des déclarations personnelles, éventuellement sujettes à des biais ou des imprécisions.  
- **Absence de variables cliniques détaillées :** Des informations médicales essentielles (comorbidités, antécédents, médicaments spécifiques) ne sont pas incluses, limitant la profondeur de l’analyse.  
- **Modèle statique :** Le dataset est transversal et ne reflète pas l’évolution des patients dans le temps, ce qui réduit la capacité à modéliser des trajectoires thérapeutiques.

##  Pistes d'amélioration  
- **Enrichissement des données :** Ajouter des variables cliniques, comportementales ou contextuelles permettrait d’obtenir un modèle plus robuste et explicatif.  
- **Collecte longitudinale :** Un suivi temporel (données hebdomadaires) permettrait d’utiliser des modèles séquentiels plus pertinents pour la santé mentale.  
- **Augmentation de la taille du dataset :** Recueillir davantage d’observations améliorerait la précision et la stabilité du modèle.  
- **Tests de modèles non linéaires :** Des algorithmes tels que Random Forest, Gradient Boosting ou XGBoost pourraient mieux capturer les interactions complexes.  
- **Techniques de régularisation :** Elles permettraient de réduire le surapprentissage et d’améliorer la généralisation.  
- **Analyse de sensibilité :** Étudier l’impact individuel de chaque variable aiderait à comprendre plus finement les facteurs influençant l’Outcome.

##  Conclusion générale  
Malgré certaines limites, le modèle développé offre une première vision intéressante de l’impact de différentes variables psychologiques et comportementales sur l’évolution du traitement. Les résultats mettent en lumière l’importance de facteurs tels que le stress, l’adhésion thérapeutique et la progression du traitement. L’intégration de données plus riches et de modèles plus avancés permettrait de renforcer la précision des prédictions et d’améliorer l’aide à la décision dans le domaine de la santé mentale.

---



