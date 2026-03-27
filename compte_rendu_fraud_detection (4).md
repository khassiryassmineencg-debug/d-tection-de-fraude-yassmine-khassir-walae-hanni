# Rapport de Travaux Pratiques
## Détection de Fraude par Apprentissage Automatique Supervisé
### Comparaison de trois méthodes de classification binaire

---

**Fichier source :** `fraud_detection_classification.ipynb`  
**Jeu de données :** `fraud_transactions_dataset.xlsx`  
**Méthodes employées :** Random Forest, Régression Logistique, Support Vector Machine  

---


---

## I. Présentation du jeu de données

Le jeu de données utilisé dans cette étude comprend 500 observations de transactions bancaires, décrites par cinq variables. Aucune valeur manquante n'a été détectée.

| Variable | Type | Description |
|:---|:---|:---|
| `Time` | Entier | Horodatage de la transaction |
| `Amount` | Réel | Montant de la transaction (en euros) |
| `Transaction_Type` | Catégoriel | Nature de la transaction : ATM, Online ou POS |
| `Location` | Catégoriel | Localisation géographique : Rural ou Urban |
| `Is_Fraud` | Binaire (0/1) | Variable cible : fraude (1) ou transaction légitime (0) |

La variable cible présente un **déséquilibre de classes marqué** : 471 transactions légitimes (94,2 %) contre 29 transactions frauduleuses (5,8 %). Ce déséquilibre constitue la principale contrainte méthodologique de l'étude.

---

## II. Analyse exploratoire des données

### II.1. Statistiques descriptives des variables numériques

| Statistique | Time | Amount |
|:---|---:|---:|
| Moyenne | 52 728 | 995,80 € |
| Écart-type | 29 284 | 576,88 € |
| Minimum | 469 | 3,41 € |
| Maximum | 99 938 | 1 984,87 € |

### II.2. Observations

L'analyse exploratoire révèle que ni la distribution des montants, ni celle des horodatages ne permettent de discriminer efficacement les transactions frauduleuses des transactions légitimes. De même, l'examen des variables catégorielles (type de transaction et localisation) n'a pas mis en évidence de catégorie concentrant massivement les fraudes. La matrice de corrélation confirme l'absence de relation linéaire forte entre les variables numériques et la variable cible.

---

## III. Prétraitement des données

Les étapes de prétraitement suivantes ont été appliquées avant l'entraînement des modèles :

**Encodage des variables catégorielles** par `LabelEncoder` :
- `Transaction_Type` : ATM → 0, Online → 1, POS → 2
- `Location` : Rural → 0, Urban → 1

**Partitionnement stratifié** des données (80 % entraînement / 20 % test) :
- Ensemble d'entraînement : 400 observations dont 23 fraudes (5,8 %)
- Ensemble de test : 100 observations dont 6 fraudes (6,0 %)

**Normalisation** par `StandardScaler`, ajusté exclusivement sur l'ensemble d'entraînement pour éviter toute fuite de données vers le test.

Le paramètre `class_weight='balanced'` a été activé pour les trois modèles afin de compenser le déséquilibre de classes lors de l'entraînement.

---

## IV. Modèles de classification

### IV.1. Random Forest

**Hyperparamètres :** 200 arbres de décision, profondeur maximale de 10, `min_samples_split = 5`, pondération des classes équilibrée.

| Métrique | Valeur |
|:---|---:|
| Accuracy | 0,9300 |
| Precision (classe fraude) | 0,0000 |
| Recall (classe fraude) | 0,0000 |
| F1-Score (classe fraude) | 0,0000 |
| AUC | 0,4743 |
| F1 moyen en validation croisée (5 folds) | 0,0000 ± 0,0000 |

Malgré une accuracy apparente de 93 %, le modèle classe l'intégralité des observations comme légitimes, ne détectant aucune fraude. Ce résultat illustre le biais induit par le déséquilibre de classes, que le paramètre `class_weight` n'a pas suffi à corriger.

---

### IV.2. Régression Logistique

**Hyperparamètres :** régularisation `C = 1,0`, solveur `lbfgs`, 1 000 itérations maximum, pondération des classes équilibrée.

| Métrique | Valeur |
|:---|---:|
| Accuracy | 0,4600 |
| Precision (classe fraude) | 0,0556 |
| Recall (classe fraude) | 0,5000 |
| F1-Score (classe fraude) | 0,1000 |
| AUC | 0,4486 |
| F1 moyen en validation croisée (5 folds) | 0,1221 ± 0,0357 |

Le modèle parvient à détecter 50 % des fraudes (3 sur 6), mais au prix d'un nombre élevé de faux positifs, ce qui explique l'accuracy dégradée à 46 %. La faiblesse de la Precision (5,56 %) traduit une discrimination insuffisante entre les deux classes.

---

### IV.3. Support Vector Machine (noyau RBF)

**Hyperparamètres :** noyau radial (RBF), `C = 1,0`, `gamma = 'scale'`, pondération des classes équilibrée.

| Métrique | Valeur |
|:---|---:|
| Accuracy | 0,6300 |
| Precision (classe fraude) | 0,0811 |
| Recall (classe fraude) | 0,5000 |
| F1-Score (classe fraude) | 0,1395 |
| AUC | 0,5514 |
| F1 moyen en validation croisée (5 folds) | 0,1175 ± 0,0480 |

Le SVM offre le meilleur compromis parmi les trois méthodes testées : il atteint le même niveau de Recall que la Régression Logistique tout en maintenant une Accuracy et une AUC supérieures. Son AUC de 0,5514 est la seule valeur légèrement au-dessus du seuil aléatoire (0,5).

---

## V. Comparaison des performances

### V.1. Tableau récapitulatif

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | **0,9300** | 0,0000 | 0,0000 | 0,0000 | 0,4743 |
| Régression Logistique | 0,4600 | 0,0556 | 0,5000 | 0,1000 | 0,4486 |
| SVM (RBF) | 0,6300 | 0,0811 | **0,5000** | **0,1395** | **0,5514** |

### V.2. Résultats de la validation croisée (5-fold stratifiée)

| Modèle | F1 moyen | Écart-type |
|:---|:---:|:---:|
| Random Forest | 0,0000 | 0,0000 |
| Régression Logistique | 0,1221 | 0,0357 |
| SVM (RBF) | 0,1175 | 0,0480 |

La validation croisée confirme l'incapacité totale du Random Forest à généraliser la détection des fraudes. La Régression Logistique et le SVM présentent des scores modestes mais non nuls, avec une instabilité notable due à la taille réduite de l'échantillon frauduleux.

### V.3. Importance des variables (Random Forest)

| Variable | Importance (Gini) |
|:---|:---:|
| Amount | ~35 % |
| Time | ~30 % |
| Transaction_Type | ~20 % |
| Location | ~15 % |

Le montant et l'horodatage de la transaction constituent les deux variables les plus informatives selon le critère d'impureté de Gini.

---

## VI. Discussion

### VI.1. Limite principale : déséquilibre de classes

Le déséquilibre extrême entre les deux classes (5,8 % de fraudes) représente l'obstacle méthodologique fondamental de cette étude. Le paramètre `class_weight='balanced'` s'est révélé insuffisant pour permettre au Random Forest d'apprendre les caractéristiques discriminantes de la classe minoritaire. Avec seulement 23 exemples de fraudes dans l'ensemble d'entraînement et 6 dans l'ensemble de test, les estimations de performance sont peu fiables et fortement dépendantes du partitionnement aléatoire des données.

### VI.2. Pistes d'amélioration

Plusieurs approches pourraient améliorer significativement les performances :

- **Rééchantillonnage par SMOTE** (*Synthetic Minority Over-sampling Technique*) : génération d'exemples synthétiques de fraudes pour équilibrer les classes dans l'ensemble d'entraînement.
- **Ajustement du seuil de décision** : abaisser le seuil de classification de 0,5 à 0,2–0,3 pour privilégier le Recall au détriment de la Precision.
- **Augmentation du volume de données** : 500 observations constituent un volume insuffisant pour un problème de détection de fraude.
- **Enrichissement des variables** : intégrer des indicateurs comportementaux tels que la fréquence de transactions, l'écart au profil habituel du client ou la vélocité géographique.
- **Modèles de gradient boosting** : XGBoost ou LightGBM offrent de meilleures garanties de robustesse face aux déséquilibres de classes.

---

## VII. Conclusion

Dans le cadre applicatif de la détection de fraude, le Recall constitue la métrique de référence : manquer une transaction frauduleuse (faux négatif) engendre des conséquences plus graves qu'un signal d'alerte intempestif sur une transaction légitime (faux positif).

À l'issue de cette comparaison, le **Support Vector Machine à noyau RBF** se distingue comme le modèle le plus performant sur l'ensemble des critères pertinents : F1-Score de 0,1395, AUC de 0,5514 et Recall de 0,5000. Le Random Forest, bien qu'affichant l'Accuracy la plus élevée (93 %), ne détecte aucune fraude et ne saurait être retenu pour un usage opérationnel.

Ces résultats demeurent cependant fragiles en raison de la taille limitée du jeu de données et du faible nombre d'exemples frauduleux. Toute mise en production devrait impérativement s'appuyer sur des techniques de rééchantillonnage et une optimisation du seuil de décision adaptée aux coûts métiers associés aux erreurs de classification.
