summary: TP1 - Nettoyer le notebook
id: tp1
categories: tp
tags: tp
status: Published
authors: OCTO Technology
Feedback Link: https://github.com/octo-technology/Formation-MLOps-1/issues/new/choose

# TP1 - Nettoyer le notebook

## Vue d'ensemble

Duration: 0:01:00

Pour réaliser ce TP, allez sur la branche `1_start_clean_notebook`
```shell
git stash
git checkout 1_start_clean_notebook
```

Dans le projet, vous trouverez :

```texte simple
    .
    └── formation_indus_ds
        ├── input
        | ├── test.csv
        | ├── train.csv
        ├── notebook
        | ├── titanic.ipynb
        README.md
```

Voici l'état d'un prototype de science des données construit par l'un de vos collègues.

## Nettoyer le notebook

Duration: 0:20:00

Le but de cette première étape est de nettoyer le notebook en suivant les pratiques de code que vous venez de
découvrir :

- Noms des variables
- Conventions
- Respecter l'ordre des cellules
- Impression
- Supprimer le code inutile
- Faire des méthodes courtes
- Respecter l'immutabilité
- Créer des fonctions

Un test pour vérifier si vous avez terminé est de redémarrer le cahier et d'exécuter toutes les cellules (symbole >>)

## Documenter les fonctions

Duration: 0:10:00

À ce stade, votre ordinateur portable est propre, il fonctionne et vous disposez de quelques fonctions.

Le but de cette étape est de documenter quelques fonctions.

Pour ce faire, vous devez utiliser les indications de type et les docstrings.

Les indications de type vous aideront à utiliser ces fonctions.

Les docstrings vous aideront à construire rapidement une bonne documentation dans les étapes suivantes.

## Extraire les méthodes dans des fichiers .py

Duration: 0:10:00

A ce stade :

- Votre ordinateur portable est propre et fonctionne
- Vous avez quelques fonctions documentées

Le but de cette étape est d'extraire vos fonctions dans un fichier `.py`.

Vous pourrez ensuite réutiliser ces fonctions dans d'autres notebooks.

Pour ce faire :

- Créer un dossier `src`.
- Créer un fichier `feature_engineering.py` dans ce dossier
- Coupez du notebook et collez dans le fichier `.py` le code de vos fonctions
- Assurez-vous d'avoir tous les `import` nécessaires.
- Pour utiliser les fonctions dans le notebook, ajoutez :

```python
import sys

sys.path.append("../src/")
from feature_engineering import *
```

## Lien vers le TP suivant

Duration: 0:01:00

Les instructions du TP suivant sont [ici](https://octo-technology.github.io/Formation-MLOps-1/tp2#0)