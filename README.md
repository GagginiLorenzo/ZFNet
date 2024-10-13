# ZFNet

ZFNet (Zeiler & Fergus Network) est un réseau de neurones convolutifs (CNN) conçu pour améliorer l'interprétabilité des modèles CNN par rapport à AlexNet. 
Ce projet implémente une version simplifié et approximative de ZFNet en utilisant TensorFlow et Keras, et inclut des méthodes pour visualiser les activations des couches à l'aide de DeconvNet.

- `base.py` : Contient les fonctions de base pour créer le modèle ZFNet, le modèle de déconvolution, et les fonctions utilitaires pour le chargement et le prétraitement des images.
- `script.py` : Script principal pour tester et visualiser les activations et les déconvolutions des couches du modèle ZFNet.
- `training.py` : Script pour entraîner le modèle ZFNet sur un ensemble de données.
- `deconve_visu/`, `exp/`, `plot/` : Répertoires pour stocker les visualisations, les expériences et les graphiques.
- `presentation/` : Contient les fichiers pour la présentation du projet.

