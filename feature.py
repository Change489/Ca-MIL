import torch
from kmeans_pytorch import kmeans
from config import NUM_CLUSTERS, DEVICE


def extract_instance_features(model, data_loader):
    """
    return
    FC_feature: [num_bags, window, ROI, ROI]
    instance_features: [num_instances, feature_dim]
    bag_labels: [num_bags]
    loss_ent
    """
    model.eval()
    FC_list = []
    instance_level_list = []
    label_list = []
    loss_ent_total = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            FC, ent_loss = model.extract_features(data)

            m_batchsize, m_channel, m_sliding, width, _ = FC.size()

            instance_mats = FC.view(m_batchsize * m_sliding, width, width)

            upper = model.upper_triangular(instance_mats)

            FC_list.append(FC)
            instance_level_list.append(upper)
            label_list.append(labels)
            loss_ent_total += ent_loss

    loss_ent_mean = loss_ent_total / len(data_loader)

    bag_labels = torch.cat(label_list, dim=0)
    FC_feature = torch.cat(FC_list, dim=0).squeeze(1)
    instance_feature = torch.cat(instance_level_list, dim=0).squeeze(1)

    return FC_feature, instance_feature, bag_labels, loss_ent_mean


def run_kmeans(instance_features):
    """
    KMeans clustering and return cluster centers
    """
    cluster_ids, cluster_centers = kmeans(
        X=instance_features.clone().detach(),
        num_clusters=NUM_CLUSTERS,
        distance='euclidean',
        device=torch.device(DEVICE)
    )
    return cluster_centers.to(DEVICE)

