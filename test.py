import torch
from config import DEVICE



def test(model,test_loader,cluster_c):
    model.eval()
    test_loss = 0.
    test_error = 0.
    ground_truth = torch.LongTensor()
    ground_truth = ground_truth.to(DEVICE)
    predicted_labels = torch.LongTensor()
    predicted_labels = predicted_labels.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        FC_list = []
        label_list = []
        for data, bag_label in test_loader:
            data, bag_label = data.to(DEVICE), bag_label.to(DEVICE)
            FC,_ = model.extract_features(data)
            m_batchsize, m_channel, m_sliding, width, _ = FC.size()
            FC_list.append(FC)
            label_list.append(bag_label)
        FC_feature = torch.cat(FC_list, dim = 0).squeeze(1)
        cluster_centers=cluster_c
        label_lis = torch.cat(label_list, dim = 0)
        predicted_label, Y_hat = model(FC_feature, cluster_centers)

        correct += (Y_hat == label_lis).sum().item()
        total += label_lis.size(0)


    acc = correct / total
    print(f"Test Accuracy = {acc:.4f}")
    return acc