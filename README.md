# ModIA_TP1

## Creer un environnement python 

python3 -m venv IA

## l'activer 

source IA/bin/activate

## désactiver 

deactivate 

# commandes d'exécution 

python train_mnist.py --epochs=10 --lr=1e-3 --batch_size=64

#afficher un tensorboard 

tensorboard --logdir runs

python mnist_app.py --weights_path [path_to_the weights]

## interprétation de la PCA

Sur la PCA/T-SNE des features (obtenir en sortie de la structure de convolution dont le but est d'extraire des features). on voit que on arrive bien à partir des features à extraire des frontières de décision entre les chiffres. 
