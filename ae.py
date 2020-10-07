import os
import sys
sys.path.append(os.path.join(os.getcwd(), "anomalydetection"))
sys.path.append(os.path.join(os.getcwd(), "datasets"))

from anomalydetection.autoencoder import AutoEncoder, Encoder
from datasets.mnist import get_dataset
from utils import save_model, convert_multimodal_mnist

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # config
    except_class = 1
    batch_size = 256
    epochs = 100

    # data
    trainset, testset = get_dataset("data")
    trainset = convert_multimodal_mnist(trainset, except_class=except_class)
    train_loader = DataLoader(trainset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)
    print("Data Load Done.")

    # model
    encoder = Encoder(input_size=28*28*1, output_size=100, n_layers=3)
    decoder = Encoder(input_size=100, output_size=28*28*1, n_layers=3)
    model = AutoEncoder(encoder, decoder).to("cuda")
    print("Model Load Done.")

    # loss & optimizer
    mse = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # training
    max_i_idx = 0
    writer = SummaryWriter("runs/ae_excls1")
    for e_idx, epoch in enumerate(range(epochs)):
        epoch_loss = []
        for i_idx, (img, label) in enumerate(train_loader):
            opt.zero_grad()

            flat_img = torch.flatten(img, start_dim=1).to("cuda")
            pred = model(flat_img)
            loss = mse(flat_img, pred)
            epoch_loss.append(loss)

            if max_i_idx < i_idx:
                max_i_idx = i_idx
            writer.add_scalar('iteration loss', loss, i_idx + (e_idx * max_i_idx))

            loss.backward()
            opt.step()
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        writer.add_scalar('epoch loss', epoch_loss, e_idx)
    writer.close()
    print("Training Done.")

    # saving
    path = os.path.join(os.getcwd(), "save_model")
    name = "ae"
    save_model(model, path, name)
    print("Saving Done.")
