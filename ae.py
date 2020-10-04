import os
import sys
sys.path.append(os.path.join(os.getcwd(), "anomalydetection"))
sys.path.append(os.path.join(os.getcwd(), "datasets"))

from anomalydetection.autoencoder import AutoEncoder, Encoder
from datasets.mnist import get_dataset
from utils import save_model

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    trainset, testset = get_dataset("data")
    train_loader = DataLoader(trainset, batch_size=256)
    test_loader = DataLoader(testset, batch_size=256)

    encoder = Encoder(input_size=28*28*1, output_size=100, n_layers=3)
    decoder = Encoder(input_size=100, output_size=28*28*1, n_layers=3)
    model = AutoEncoder(encoder, decoder).to("cuda")

    mse = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10
    training_loss = []
    trained_loss = []
    for img, label in train_loader:
        flat_img = torch.flatten(img, start_dim=1).to("cuda")
        pred = model(flat_img)
        loss = mse(flat_img, pred)
        training_loss.append(loss)

    for idx, epoch in enumerate(range(epochs)):
        for img, label in train_loader:
            opt.zero_grad()
            flat_img = torch.flatten(img, start_dim=1).to("cuda")
            pred = model(flat_img)
            loss = mse(flat_img, pred)
            loss.backward()
            opt.step()

    for img, label in train_loader:
        flat_img = torch.flatten(img, start_dim=1).to("cuda")
        pred = model(flat_img)
        loss = mse(flat_img, pred)
        trained_loss.append(loss)

    print("학습 전 : {}".format(sum(training_loss) / len(training_loss)))
    print("학습 후 : {}".format(sum(trained_loss) / len(trained_loss)))

    path = os.path.join(os.getcwd(), "save_model")
    name = "ae"
    save_model(model, path, name)
