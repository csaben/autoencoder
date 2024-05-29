import logging
import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from autoencoder.ae import AutoEncoder
from autoencoder.transforms import get_dataset, load

# Setup default logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/detecting_anomalies_asgroup.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and optimizer states
model = AutoEncoder()
model.to(device=device)  # Move model to the specified device
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
load("autoencoder.pth", model, optimizer)

# Load anomalies and uninfected datasets
load_dotenv("data.env")

infected_ds = get_dataset(os.getenv("infected_patients_directories"))
uninfected_ds = get_dataset(os.getenv("uninfected_patients_directories"))
data_loader_in = DataLoader(infected_ds, batch_size=32, shuffle=True, num_workers=4)
data_loader_un = DataLoader(uninfected_ds, batch_size=32, shuffle=True, num_workers=4)

# Set threshold for loss and see if anomalies are detected
# criterion = nn.MSELoss(reduction="none")  # Use 'none' to get individual losses
criterion = nn.MSELoss()


def anomaly_threshold(loss, threshold=0.0045):
    """Return True if anomaly"""
    return loss > threshold


for img_in, img_un in zip(data_loader_in, data_loader_un):
    img_in, img_un = img_in.to(device), img_un.to(device)  # Move images to the device

    # size of batch torch.Size([B, 3, 128, 128]), 3 b/c RGB
    recon_in = model(img_in)
    recon_un = model(img_un)
    loss_in = criterion(recon_in, img_in)  # Get individual losses
    loss_un = criterion(recon_un, img_un)  # Get individual losses

    # Aggregate losses for each image in the batch
    loss_in_per_image = loss_in.view(loss_in.size(0), -1).mean(dim=1)
    loss_un_per_image = loss_un.view(loss_un.size(0), -1).mean(dim=1)

    for i, (loss_img_in, loss_img_un) in enumerate(
        zip(loss_in_per_image, loss_un_per_image)
    ):
        LOGGER.info(
            f"Infected image {i} registered as an anomaly: {anomaly_threshold(loss_img_in.item())} with loss {loss_img_in.item()}"
        )
        LOGGER.info(
            f"Uninfected image {i} registered as an anomaly: {anomaly_threshold(loss_img_un.item())} with loss {loss_img_un.item()}"
        )
