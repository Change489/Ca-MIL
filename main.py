
from dataset import get_dataloaders
from model import CaMIL
from train import train
from test import test
from config import ROI_NUM, NUM_WINDOWS, TIMEPOINTS, DEVICE


def main():

    # create DataLoader
    train_loader, test_loader = get_dataloaders()

    # 创建模型
    model = CaMIL(
        num_windows=NUM_WINDOWS,
        ROI_num=ROI_NUM,
        timepoints=TIMEPOINTS,
        device=DEVICE
    ).to(DEVICE)

    # train
    cluster_centers = train(model, train_loader)
    # test
    test(model, test_loader, cluster_centers)


if __name__ == "__main__":
    main()
