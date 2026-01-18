# Projet Explaining and Harnessing Adversarial Examples 2015

À travers différentes architectures (softmax, maxouts, rbf, ensemble), ce dépôt teste les limites de la robustesse des réseaux de neurones face à des perturbations calculées, démontrant que même des modèles très performants ou complexes peinent à se défendre contre des attaques optimisées.

## Liste des fichiers

### `4-linear perturbations for non linear models`
Description : Dossier contenant une autre implémentation des essais de vulnérabilité sur les différents modèles.

### `old`
Description : Dossier contenant un ensemble de fichiers issus des différents essais sur l'implémentation de l'article.

### 0. `0.googlenet.ipynb`
Description : Utilisation d'un modèle GoogleNet pré-entraîné pour générer des exemples adversariaux concrets.
```bash
jupyter notebook 0.googlenet.ipynb
```
### 1. `1.softmax.ipynb`
Description : Entraînement d'un réseau softmax et essais de vulnérabilité, affichage des poids du réseau et essais sur des exemples rubbish.
```bash
jupyter notebook 1.softmax.ipynb
```
### 2. `2.maxouts.ipynb`
Description : Implémentation et entraînement de réseaux utilisant des couches Maxout au lieu des activations traditionnelles (ReLU/Sigmoid) pour tester si l'optimisation par morceaux offre une meilleure résistance.
```bash
jupyter notebook 2.maxouts.ipynb
```
### 3. `3.convnet_cifar.ipynb`
Description : Entraînement d'un réseau convolutionnel et essais de vulnérabilité, essais sur des exemples rubbish.
```bash
jupyter notebook 3.convnet_cifar.ipynb
```
### 4. `rbf.ipynb`
Description : Entraînement d'un réseau RBF et essais de vulnérabilité, essais sur des exemples rubbish, tests de transfert (softmax vers RBF)
```bash
jupyter notebook 4.rbf.ipynb
```
### 5. `5.mpdbm.ipynb`
Description : Entraînement d'un Proxy MP-DBM (Multi-Prediction Deep Boltzmann Machine) pour voir si contraindre le modèle à apprendre la distribution générative des données améliore sa robustesse.
```bash
jupyter notebook 5.mpdbm.ipynb
```
### 6. `6.ensemble(12maxouts).ipynb`
Description : Entraînement conjoint de 12 modèles Maxout pour vérifier si la moyenne des prédictions permet de diluer l'effet des perturbations adversariales.
```bash
jupyter notebook 6.ensemble(12maxouts).ipynb
```
### 7. `Approfondissement_article.ipynb`
Description : à remplir
```bash
jupyter notebook Approfondissement_article.ipynb
```
