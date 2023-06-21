import argparse
from config import get_config
# from models.dat import build_model
import torch
from models.dat import DAT
import torch.optim as optim 
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def train(model, tr_loader, val_loader, optimizer, criterion, max_epoch, print_per_iter):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    writer = SummaryWriter()
    model = model.to(device)
    count = 0
    for ep in range(max_epoch):
        model.train()

        iter_num = 0
        for data, label in tr_loader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = criterion(out, torch.nn.functional.one_hot(label,1000).to(torch.float))
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss", loss,count)

            iter_num += 1
            count += 1
            if iter_num % print_per_iter == 0:
                print(f'Epoch {ep:4d} Iter {iter_num:6d} ---> Loss: {loss}')
            


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DAT(224, 1000).to(device=device)
    model.train()
    x = torch.randn(4,3,224,224).to(device=device)
    x.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)
    criterion = torch.nn.BCELoss()
    

    tr_trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder('/home/ubuntu/Imagenet/imagenet/train', tr_trans)
    # dataset = datasets.ImageNet(root='/home/ubuntu/Imagenet/imagenet', split='train', transform=tr_trans)

    dataloder = DataLoader(dataset=dataset, shuffle = True, batch_size=16)

    train(model=model, tr_loader=dataloder, val_loader=None, optimizer=optimizer, criterion=criterion, max_epoch=100, print_per_iter=5)

    for data, label in dataloder:
        # data = data.reshape()
        data = data.to(device)
        label = label.to(device)
        out = model(data) 
        # plt.imshow(data[1].reshape(data.shape[1:4]).permute(1,2,0))
        # plt.show()
        print()





    orig_label = torch.zeros(4,1000)
    orig_label[:,44] = 1
    # orig_label[:,44] = 1
    temp = 0
    for epoch in range(1000):
        while(True):
            optimizer.zero_grad()
            out = model(x)
            # fake_out = torch.zeros(10,1000)
            # fake_out[:,44] = 1
            loss = criterion(out, orig_label)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss)
            print("sum: ", torch.sum(out))



if __name__ == "__main__":
    main()