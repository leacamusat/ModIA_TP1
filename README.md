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


## Gcloud comment copier un dossier dans le repository de Google Cloud, pour lancer un Docker

1) activer l'instance sur Gcloud 

OUVRIR UN TERMINAL EN LOCAL

2) copier la commande avec ssh dans un terminal (apparait en vert, on est sur la machine distante)

gcloud compute ssh --zone "europe-west1-b" "lea"  --project "ai-frameworks-2022-2023"

On est maintenant sur le terminal à distance 

mkdir Workspace 

cd Workspace 

OUVRIR UN AUTRE TERMINAL LOCAL 

3)sur un terminal LOCAL gcloud compute scp --recurse ModIA_TP1 lea:Workspace/ --zone "europe-west1-b" ##copier tout le repositoy à distance 

SUIVRE LE TP DANS LE TERMINAL DISTANT

#créer une image 

sudo docker build -t image_s1 .
