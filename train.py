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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def train(model, tr_loader, val_loader, optimizer, criterion, max_epoch, print_per_iter, val_per_iter):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    writer = SummaryWriter()
    model = model.to(device)
    count = 0
    best_val_acc = 0
    for ep in range(max_epoch):
        model.train()

        iter_num = 0
        for data, label in tr_loader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            # loss = criterion(out, torch.nn.functional.one_hot(label,1000).to(torch.float))
            loss = criterion(out,label)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss,count)

            iter_num += 1
            count += 1
            if iter_num % print_per_iter == 0:
                print(f'Epoch {ep:4d} Iter {iter_num:6d} ---> Loss: {loss}')
            
            if iter_num % val_per_iter == 0:
                val_loss = 0
                val_acc = 0
                val_total = 0
                model.eval()
                with torch.no_grad():
                    for val_data, val_label in val_loader:
                        val_data = val_data.to(device)
                        val_label = val_label.to(device)
                        val_out = model(val_data)
                        val_loss += criterion(val_out, val_label).item()
                        val_acc += (val_out.argmax(dim=1)==val_label).sum().item()
                        val_total += val_label.shape[0]
                val_loss /= len(val_loader)
                val_acc = 100 * (val_acc / val_total)
                print(f'Val Loss: {val_loss}   Val Acc: {val_acc}')
                writer.add_scalar("Loss/val", val_loss, count)
                writer.add_scalar("Acc/val", val_acc, count)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint = {
                        'epoch' : ep + 1,
                        'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'val_acc' : val_acc
                    }
                    torch.save(checkpoint, "checkpoint_best.pth.tar")

                model.train()
    writer.close()




def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DAT(224, 1000).to(device=device)
    model.train()
    # x = torch.randn(4,3,224,224).to(device=device)
    # x.requires_grad = True

    # optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)
    # criterion = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr = 0.001)

    tr_trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # tr_dataset = datasets.ImageFolder('/home/ubuntu/Imagenet/low-resolution', tr_trans)
    # val_dataset = datasets.ImageFolder('/home/ubuntu/Imagenet/low-resolution', tr_trans)
    tr_dataset = datasets.ImageFolder('/home/ubuntu/Imagenet/imagenet/ILSVRC/Data/CLS-LOC/train', tr_trans)
    val_dataset = datasets.ImageFolder('/home/ubuntu/hacettepe/dat/DeformableTransformer/val_foldered', tr_trans)


    tr_dataloder = DataLoader(dataset=tr_dataset, shuffle = True, batch_size=64)
    val_dataloder = DataLoader(dataset=val_dataset, shuffle = True, batch_size=64)

    train(model=model, tr_loader=tr_dataloder, val_loader=val_dataloder, optimizer=optimizer, criterion=criterion, max_epoch=150, print_per_iter=400, val_per_iter=int(len(tr_dataloder)/10))

    # for data, label in tr_dataloder:
    #     # data = data.reshape()
    #     data = data.to(device)
    #     label = label.to(device)
    #     out = model(data) 
    #     # plt.imshow(data[1].reshape(data.shape[1:4]).permute(1,2,0))
    #     # plt.show()
    #     print()





    # orig_label = torch.zeros(4,1000)
    # orig_label[:,44] = 1
    # # orig_label[:,44] = 1
    # temp = 0
    # for epoch in range(1000):
    #     while(True):
    #         optimizer.zero_grad()
    #         out = model(x)
    #         # fake_out = torch.zeros(10,1000)
    #         # fake_out[:,44] = 1
    #         loss = criterion(out, orig_label)
    #         loss.backward()
    #         optimizer.step()
    #         print("Loss: ", loss)
    #         print("sum: ", torch.sum(out))



if __name__ == "__main__":
    main()