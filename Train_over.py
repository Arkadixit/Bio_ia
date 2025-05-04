import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from cnn_over import Net
import time
import matplotlib.pyplot as plt
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def tracer_courbe(x_1,x_2,x_3,x_4, y, titre="Courbe", xlabel="Epochs", ylabel="Loss",fichier=None):
    if len(x_1) != len(y) or len(x_2)!=len(y):
        raise ValueError("Les listes x et y doivent avoir la même longueur.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(y,x_1, marker='o',label="Training Loss Curve",color = "green")  
    plt.plot(y, x_2, marker='x',label="Validation Loss Curve",color="red" )
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(y,x_4, marker='o',label="Training Accuracy",color = "green")  
    plt.plot(y,x_4, marker='x',label="Validation Accuracy",color="red" )
    plt.title("Accuracy")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy %")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    nom_fichier = "Figures/"+ fichier
    if nom_fichier:
        plt.savefig(nom_fichier, dpi = 300)

    plt.show()


def train_one_epoch(epoch_index, tb_writer, train_loader, net, criterion, optimizer, device):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 314 == 313:
            last_loss = running_loss / 314
            print(f'batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def main():
    writer = SummaryWriter('bim/dogs-vs-cats-logs')

    train_loss_curv = []
    val_loss_curv = []
    Epochs = []


    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 3
    data_dir = 'Data'
    train_dataset = torchvision.datasets.ImageFolder(data_dir + '/Training', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = torchvision.datasets.ImageFolder(data_dir + '/Validation', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = {0: 'glioma', 1: 'meningioma', 3: 'notumor', 4: 'pituitary'}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    EPOCHS = 30
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        Epochs.append(epoch+1)
        start_time = time.time()
        print(f'EPOCH {epoch + 1}:')
        net.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_loader, net, criterion, optimizer, device)

        running_vloss = 0.0
        net.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = net(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        writer.add_scalars('Training vs. Validation Loss',
                           { 'Training': avg_loss, 'Validation': avg_vloss },
                           epoch + 1)
        writer.flush()

        train_loss_curv.append(avg_loss)
        val_loss_curv.append(avg_vloss)

        if(avg_vloss < best_vloss):
            best_vloss = avg_vloss
            model_path = f'model/model_{timestamp}_{epoch}'
            torch.save(net.state_dict(), model_path)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"il a mis {total_time} secondes")

        epoch_number += 1

    train_accuracy = []
    val_accuracy = []

    for n in range(len(train_loss_curv)):
        train_accuracy.append(100*math.exp(-train_loss_curv[n]))
        val_accuracy.append(100*math.exp(-val_loss_curv[n]))

    tracer_courbe(train_loss_curv,val_loss_curv,train_accuracy,val_accuracy, Epochs, titre="Loss Curve", xlabel="Epochs", ylabel="Loss", fichier = "Loss Test")
    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    main()
