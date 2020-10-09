import os
import sys
sys.path.append(os.path.join(os.getcwd(), "anomalydetection"))
sys.path.append(os.path.join(os.getcwd(), "datasets"))

from anomalydetection.autoencoder import AutoEncoder, Encoder
from datasets.mnist import get_dataset
from utils import save_model, convert_multimodal_mnist, show_online_image

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # 모든 class에 대해 실험
    for except_class in list(range(10)):
        # config
        # except_class = 1
        batch_size = 256
        epochs = 100
        bottleneck_size = 100
        n_layers = 5
        learning_rate = 0.001
        config_summary_name = "excls"+str(except_class) + \
                              "_nlayers"+str(n_layers)

        # data
        trainset, testset = get_dataset("data")
        trainset = convert_multimodal_mnist(trainset, except_class=except_class)
        showing_train_loader = DataLoader(show_online_image(trainset, except_class=except_class, amount=10), batch_size=90)
        showing_test_loader = DataLoader(show_online_image(testset, except_class=None, amount=10), batch_size=100)
        train_loader = DataLoader(trainset, batch_size=batch_size)
        test_loader = DataLoader(testset, batch_size=batch_size)
        print("Data Load Done.")

        # model
        encoder = Encoder(input_size=28*28*1, output_size=bottleneck_size, n_layers=n_layers)
        decoder = Encoder(input_size=bottleneck_size, output_size=28*28*1, n_layers=n_layers)
        model = AutoEncoder(encoder, decoder).to("cuda")
        print("Model Load Done.")

        # loss & optimizer
        mse = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training
        max_i_idx = 0
        writer = SummaryWriter("runs/ae/"+config_summary_name)
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

            # 중간 이미지 보기
            if (e_idx+1) % 10 == 0:
                for s_idx, (img, label) in enumerate(showing_train_loader):
                    flat_img = torch.flatten(img, start_dim=1).to("cuda")
                    pred = model(flat_img)
                    grid = torchvision.utils.make_grid(pred.reshape(-1,1,28,28).detach().to("cpu"), nrow=10)
                    writer.add_image("epochs"+str(e_idx+1)+"_train_images", grid, 0)

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            writer.add_scalar('epoch loss', epoch_loss, e_idx)

        print("Training Done.")

        # saving
        path = os.path.join(os.getcwd(), "save_model")
        name = "ae_" + config_summary_name
        save_model(model, path, name)
        print("Saving Done.")

        # test 이미지 보기
        for img, label in showing_test_loader:
            flat_img = torch.flatten(img, start_dim=1).to("cuda")
            pred = model(flat_img)
            grid = torchvision.utils.make_grid(pred.reshape(-1, 1, 28, 28).detach().to("cpu"), nrow=10)
            writer.add_image("test_images", grid, 0)

        writer.close()

        # testing
        score = []
        y_true = []
        for img, label in test_loader:
            flat_img = torch.flatten(img, start_dim=1).to("cuda")
            pred = model(flat_img)

            score += ((flat_img-pred)**2).sum(axis=1).detach().to("cpu").tolist()
            y_true += np.where(label==except_class, 1, 0).tolist()

        print("auroc : {}".format(roc_auc_score(y_true, score)))

