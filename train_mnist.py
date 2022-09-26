import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnist_net import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, writer, epochs=10): 
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs): #pour chaque epoch 
        running_loss = []
        t = tqdm(loader) #barre d'avancée d'exécution
        for x, y in t: #on prend batch par batch 
            x, y = x.to(device), y.to(device) #envoyer sur le gpu ou cpu(device)
            outputs = net(x) #réseau qu'on a défini précédemment sur l'entrée x 
            loss = criterion(outputs, y) #fonction perte entre la préditction (output) et la sortie réelle
            running_loss.append(loss.item()) #ajout de la valeur de la fonction perte 
            optimizer.zero_grad() #met à 0 les différents gradients (pour tous les batchs) 
            loss.backward() #backward phase
            optimizer.step() #Performs a single optimization step (parameter update).
            t.set_description(f'training loss: {mean(running_loss)}') #affichage 
        writer.add_scalar("training loss", mean(running_loss), epoch)

def test(net, dataloader):
    test_corrects = 0 #
    total = 0
    with torch.no_grad(): #on recalcule pas les gradients pendant le test
        for x, y in dataloader: # chargement des données
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x).argmax(1) #softmax : la sortie du RN est celle qui a la proba la plus élevée
            test_corrects += y_hat.eq(y).sum().item() #compte le nb de predictions correctes
            # .item() Returns the value of this tensor as a standard Python number.(convertion en int)
            total += y.size(0) #nombre d'éléments dans le batch 
    return test_corrects / total #pourcentage total de bonnes prédictions 

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  #instanciation d'un parseur qui permet la saisie de données par l'utilisateur 
  
  #ajout d'un nouvel argument en précisant son nom, son type, une valeur par défaut et message d'aide
  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size', type=int, default =32, help='size of batch')
  parser.add_argument('--lr', type=float, default =1e-2, help='learning rate')
  parser.add_argument('--epochs', type=int, default =50, help='number of epochs')

  args = parser.parse_args()
  exp_name = args.exp_name
  epochs = args.epochs
  batch_size = args.batch_size
  lr = args.lr

  # transforms
  transform = transforms.Compose( #transform : activer les operations disponibles sur les images
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))]) 
      #transformations en tenseur et normalisation 
      #a tensor image with mean and standard deviation

  # datasets
  #chargement des images + transformations en tenseur et normalisation des images
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  #chargement en batch de taille size batch en mélangeant avec num_workers= 2 cad au plus deux workers peuvent mettre les données dans la ram
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  
  net = MNISTNet() #instanciation du modèle
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=lr) #les paramètres sont les poids et biais et sont implicites
  
  writer = SummaryWriter(f'runs/MNIST')
  train(net, optimizer, trainloader, writer, epochs)
  test_acc = test(net, testloader)
  print(f'Test accuracy:{test_acc}')
  torch.save(net.state_dict(), "mnist_net.pth")
  #add embeddings to tensorboard
  perm = torch.randperm(len(trainset.data)) 
  images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
  images = images.unsqueeze(1).float().to(device)
  with torch.no_grad():
    embeddings = net.get_features(images)
    writer.add_embedding(embeddings,
                  metadata=labels,
                  label_img=images, global_step=1)

  # save networks computational graph in tensorboard
  writer.add_graph(net, images)
  # save a dataset sample in tensorboard
  img_grid = torchvision.utils.make_grid(images[:64])
  writer.add_image('mnist_images', img_grid)


  #fn : fonction qui met une interface utilisateur sur le site web
  #input : text, audio ou image
  #output : test, image, label
  
  

  ##lancer la commande suivant en premier dans 
  # tensorboard --logdir runs

#python train_mnist.py --epochs=10 --lr=1e-3 --batch_size=64