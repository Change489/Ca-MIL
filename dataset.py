import torch
from torch.utils.data import DataLoader, TensorDataset
from config import ROI_NUM, NUM_WINDOWS, TIMEPOINTS, BATCH_SIZE


def create_dummy_dataset(num_train=20, num_test=5):

    x_train = torch.rand(num_train, NUM_WINDOWS, ROI_NUM, TIMEPOINTS)
    y_train = torch.randint(0, 2, (num_train,))
    x_test = torch.rand(num_test, NUM_WINDOWS, ROI_NUM, TIMEPOINTS)
    y_test = torch.randint(0, 2, (num_test,))

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset


def get_dataloaders():

    train_dataset, test_dataset = create_dummy_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader
