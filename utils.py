import numpy as np
import time
import torch
import random
import sys
import os
import glob
device = 'cuda:0'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
def time_now():

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def print_config(config_dict):

    print("\n:")
    for k, v in config_dict.items():
        print(f"{k}: {v}")
    print("")

def calculate_metrics_and_plot_roc(pred, labels):

    pred_labels = torch.ge(pred, 0.5).float()

    pred_labels_np = pred_labels.cpu().numpy()
    true_labels_np = labels.cpu().numpy()

    tn, fp, fn, tp = confusion_matrix(true_labels_np, pred_labels_np).ravel()

    acc = accuracy_score(true_labels_np, pred_labels_np)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    pre = tp / (tp + fp)
    f1 = 2*pre*sen/(pre+sen)
    auc = roc_auc_score(true_labels_np, pred.cpu().numpy())

    # plt.figure()
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # plt.show()

    return acc, sen, spe, pre, f1, auc

def rearrange_fit_L_New(timeSeries,windows_length=30,windows_gap =30):
    timeSeries_shape = timeSeries.shape
    windows_num = int((timeSeries_shape[2] - windows_length) / windows_gap) + 1

    data = timeSeries[:, :, (0 * windows_gap):(windows_length + 0 * windows_gap)]
    data = data.reshape(timeSeries_shape[0], 1, 1, timeSeries_shape[1], windows_length)
    for j in range(1,windows_num):
        A = timeSeries[:, :, (j * windows_gap):(windows_length + j * windows_gap)]
        A = A.reshape(timeSeries_shape[0], 1,1, timeSeries_shape[1], windows_length)
        data = np.concatenate((data,A),axis=2)


    return data


def seed_it(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
def GetPearson(matrix):
    num_samples, num_rows, num_cols = matrix.size()
    output_matrix = torch.zeros(num_samples, num_rows, num_rows)

    for i in range(num_samples):
        sample = matrix[i]
        sample = sample.view(num_rows, -1)

        output = torch.corrcoef(sample)
        output = torch.where(torch.isnan(output), 0, output)
        output =output.to(device)

        mask = torch.eye(n = num_rows).to(device)
        output = mask + (1 - mask) * output

        output_matrix[i] = output
    return output_matrix.to(device)


def save_model(model, optimizer, epoch, loss, save_path):

    folder = os.path.dirname(save_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(state, save_path)
    print("saveto", save_path)

def load_model(model, optimizer, load_path, device="cpu"):

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint.get("epoch", None)
    loss = checkpoint.get("loss", None)

    print("loaded", load_path, "finish")
    return model, optimizer, epoch, loss