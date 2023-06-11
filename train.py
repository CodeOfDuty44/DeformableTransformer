import argparse
from config import get_config
# from models.dat import build_model
import torch
from models.dat import DAT
import torch.optim as optim 
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
def main():

    model = DAT(224, 1000).to(device="cpu")
    model.train()
    x = torch.randn(4,3,224,224).to(device="cpu")
    x.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr = 0.1)
    criterion = torch.nn.BCELoss()
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

    # dataset = datasets.ImageNet(root='path/to/imagenet', split='train')
    # dataloder = DataLoader(dataset=dataset, shuffle = True, batch_size=4)
    # for data, label in dataloder:
    #     data = data.reshape()
    #     out = model(data)


if __name__ == "__main__":
    main()