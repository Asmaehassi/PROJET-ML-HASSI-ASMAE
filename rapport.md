#  **Projet Machine Learning — Analyse de la Santé Mentale**
**Étudiante : Asmae Hassi**  
**Module : Data Science / Machine Learning**    


---

#  **1. Introduction**

Cette étude consiste à analyser un dataset portant sur des patients en suivi psychiatrique, afin de prédire l’issue d’un traitement mental (amélioration ou non).  
L’objectif est de transformer des données brutes en informations exploitables et en modèle prédictif performant.  .

La thématique choisie est :  
 **Santé : Analyse de données liées à la santé mentale.**

## 🧾 2. Description du Dataset

Le dataset *Mental Health Diagnosis and Treatment Monitoring* contient **500 lignes et 17 colonnes**.  
Les données décrivent des patients, leurs symptômes, leur traitement et l’issue observée. :contentReference[oaicite:0]{index=0}

### 📊 Structure des données  


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

---

##  3. Prétraitement des données

Les étapes principales de nettoyage et de prétraitement ont été :

- **Suppression des doublons**
- **Gestion des valeurs manquantes**
- **Encodage des variables catégorielles**, notamment `gender` et `outcome`
- **Standardisation des variables numériques** pour faciliter l’apprentissage des modèles
- **Séparation en ensembles d’entraînement (80%) et de test (20%)**

---
#  1.Code utilisé et description
Dans cette section, nous présentons l’ensemble des blocs de code développés pour mener à bien le projet de Machine Learning. Chaque portion de code est accompagnée d’une brève description permettant de comprendre son rôle dans la chaîne de traitement : préparation des données, exploration, modélisation, optimisation et évaluation.
L’objectif est d’exposer clairement la démarche méthodologique suivie et de justifier les choix techniques réalisés, conformément au cahier des charges.
Description :
Ce bloc importe toutes les librairies nécessaires pour :

Manipuler les données (Pandas, Numpy)

Faire des graphiques (Matplotlib, Seaborn)

Faire le prétraitement (encodage, scaling)

Tester plusieurs modèles de Machine Learning

Mesurer la performance (accuracy, classification report)

 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
