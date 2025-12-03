import torch
import torch.optim as optim
from config import DEVICE, EPOCHS, LR
from feature import extract_instance_features, run_kmeans


def train(model, train_loader):

    optimizer = optim.Adam(model.parameters(), lr = LR)

    for epoch in range(EPOCHS):
        model.train()


        FC_feature, instance_feature, bag_labels, loss_consistency = \
            extract_instance_features(model, train_loader)

        cluster_centers = run_kmeans(instance_feature)

        pred, Y_hat = model(FC_feature.to(DEVICE), cluster_centers)

        bce = torch.nn.BCELoss()(pred, bag_labels.float().to(DEVICE))
        loss = bce + loss_consistency


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss={loss.item():.4f}")


    return cluster_centers

