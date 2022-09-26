import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utiliser un tensorboard :
from torch.utils.tensorboard import SummaryWriter

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        #il y a autant de filtres de convolution que de canaux de sortie de la convolution précédente (comme des sous images)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3)) 
        
        #prend en arg le nombre de canaux d'entrée, le nombre de canaux de sortie et la taille du filtre de convolution
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        #prend en argument la taille du filtre de pooling et le stride c'est de combien de pixels on décale le filtre à chaque étape
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.flatten=nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by, relu is in F (torch.nn.functional.relu)
        x = self.pool(x)                # a relu activation and a max pooling#
        x = F.relu(self.conv2(x)) 
        x = self.pool(x)   
        x = self.flatten(x)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x

    def get_features(self, x): #
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4) #idem que flatten
        return x
    
if __name__=='__main__':
    x = torch.rand(16,1,28,28)
    net = MNISTNet()
    y = net(x)
    assert y.shape == (16,10)
