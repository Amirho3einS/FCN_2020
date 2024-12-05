import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import medfilt
from importlib.resources import files


# Costum Dataset
class EEGDataset(Dataset):
    def __init__(self, data, fs=256, window_size=4, overlap=1):
        self.data = data
        self.fs = int(fs)
        self.window_size = int(window_size)
        self.overlap = int(overlap)

    def __len__(self):
        data_len = self.data.shape[1]
        return ((data_len // self.fs) - self.window_size) // self.overlap + 1

    def __getitem__(self, idx):
        start = idx * self.overlap * self.fs
        end = start + self.window_size * self.fs
        return self.data[:, start:end]


class FCN2(nn.Module):
    def __init__(self, in_channels=18):
        super(FCN2, self).__init__()
        # first convolutional block
        n_filters = 128
        self.conv1 = nn.Conv1d(
            in_channels, n_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        # first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4, padding=0)

        # second convolutional block
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()
        # second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4, padding=0)

        # third convolutional block
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.relu3 = nn.ReLU()
        # second pooling
        self.pool3 = nn.MaxPool1d(kernel_size=4, padding=0)

        # Fully connected layers within the classifier sequential module
        self.classifier = nn.Sequential(
            nn.Conv1d(n_filters, 100, kernel_size=16, padding=0),
            nn.Conv1d(100, 2, kernel_size=1, padding=0),
        )

    def get_features(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        return out

    def forward(self, x):
        features = self.get_features(x)
        out = self.classifier(features)
        out = out.transpose(0, 1)  # nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx(t*b)
        out = out.t()  # (t*b)xn

        return out


def fcn_algorithm(
    eeg_data,
    fs=256,
    window_size=4,
    overlap=1,
    filter_size=11,
):
    """
    Gotman algorithm for seizure detection
    Args:
    t: array of time points
    eeg_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    window_size: size of the window in seconds
    overlap: overlap between windows in seconds
    seizure_detections: boolean array of seizure detections, shape (n_samples,)

    **NOTE**: Other than default values has not been tested
    """

    eeg_dataset = EEGDataset(
        eeg_data, fs=fs, window_size=window_size, overlap=overlap
    )

    # DataLoader
    batch_size = 64
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=False)

    model_path = files("fcn_2020").joinpath("model_weights.pth")

    # Load the model
    model = FCN2()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["state_dict"]
    )

    # Set the model to evaluation mode
    model.eval()

    out = []
    for data in dataloader:
        data = data.float().to(device)
        out.append(model(data))
    out = torch.cat(out)

    preds = out.argmax(dim=1).cpu().numpy()
    # Inserting the first and last 2 seconds of the data
    # 1 the beginning of the data
    preds = np.insert(preds, 0, [0])
    # 2 the end of the data
    preds = np.insert(preds, -1, [0, 0])

    pred_labels_smooth = medfilt(preds, kernel_size=filter_size)

    return pred_labels_smooth
