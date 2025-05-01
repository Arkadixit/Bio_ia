import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from cnn import Net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        if i % 2000 == 1999:
            last_loss = running_loss / 1000
            print(f'batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def main():
    writer = SummaryWriter('bim/dogs-vs-cats-logs')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 10
    data_dir = 'archive'
    train_dataset = torchvision.datasets.ImageFolder(data_dir + '/Train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = torchvision.datasets.ImageFolder(data_dir + '/Val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = {0: 'Glioma', 1: 'Meningioma', 3: 'No Tumor', 4: 'Pituitary'}

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

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model/model_{timestamp}_{epoch}'
            torch.save(net.state_dict(), model_path)

        epoch_number += 1

    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    main()
