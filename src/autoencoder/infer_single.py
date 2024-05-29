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
        logging.FileHandler("logs/detecting_anomalies_latest.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and optimizer states
model = AutoEncoder().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
load("autoencoder.pth", model, optimizer)

# Load anomalies and uninfected
load_dotenv("data.env")

infected_ds = get_dataset(os.getenv("infected_patients_directories"))
uninfected_ds = get_dataset(os.getenv("uninfected_patients_directories"))
root_dir = os.getenv("infected_patients_directories")
data_loader_in = DataLoader(infected_ds, batch_size=32, shuffle=True, num_workers=4)
data_loader_un = DataLoader(uninfected_ds, batch_size=32, shuffle=True, num_workers=4)

# Set threshold for loss and see if anomalies are detected
criterion = nn.MSELoss()


def anomaly_threshold(loss):
    """Return True if anomaly"""
    return loss > 0.018  # determined via observation


model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for img_in, img_un in zip(data_loader_in, data_loader_un):
        img_in = img_in.to(device)
        img_un = img_un.to(device)

        for i in range(img_in.size(0)):
            single_img_in = img_in[i].unsqueeze(
                0
            )  # add a batch dimension of (1, ...) so network can handle it
            single_img_un = img_un[i].unsqueeze(0)  # add a batch dimension of (1, ...)

            recon_in = model(single_img_in)
            recon_un = model(single_img_un)

            loss_in = criterion(recon_in, single_img_in)
            loss_un = criterion(recon_un, single_img_un)

            LOGGER.info(
                f"infected image {i} registered as an anomaly: {anomaly_threshold(loss_in.item())} : loss: {loss_in.item()}"
            )
            LOGGER.info(
                f"uninfected image {i} registered as an anomaly: {anomaly_threshold(loss_un.item())} : loss: {loss_un.item()}"
            )
