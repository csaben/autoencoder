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
        logging.FileHandler("logs/detecting_anomalies.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model and optimizer states
model = AutoEncoder().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
load("autoencoder.pth", model, optimizer)

# load anomalies and uninfected

load_dotenv("data.env")

infected_ds = get_dataset(os.getenv("infected_patients_directories"))
uninfected_ds = get_dataset(os.getenv("uninfected_patients_directories"))
root_dir = os.getenv("infected_patients_directories")
data_loader_in = DataLoader(infected_ds, batch_size=32, shuffle=True, num_workers=4)
data_loader_un = DataLoader(uninfected_ds, batch_size=32, shuffle=True, num_workers=4)

# set threshold for loss and see if anomalies are detected
criterion = nn.MSELoss()


def anomaly_threshold(loss):
    """return True if anomaly"""
    # return loss > 0.0045
    return loss > 0.0


model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for img_in, img_un in zip(data_loader_in, data_loader_un):
        # size of batch torch.Size([B, 3, 128, 128]), 3 b/c RGB
        img_in = img_in.to(device)
        img_un = img_un.to(device)
        recon_in = model(img_in)
        recon_un = model(img_un)
        loss_in = criterion(recon_in, img_in)
        loss_un = criterion(recon_in, img_in)

        LOGGER.info(
            f"infected image registered as an anomaly: {anomaly_threshold(loss_in.item())} : loss: {loss_in.item()} "
        )
        LOGGER.info(
            f"uninfected image registered as an anomaly: {anomaly_threshold(loss_un.item())} : loss: {loss_un.item()} "
        )
