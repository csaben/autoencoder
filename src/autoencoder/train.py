import logging
import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from autoencoder.ae import AutoEncoder
from autoencoder.transforms import get_dataset, load, save

# Setup default logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

load_dotenv("data.env")

uninfected_ds = get_dataset(os.getenv("uninfected_patients_directories"))
root_dir = os.getenv("infected_patients_directories")
data_loader_un = DataLoader(uninfected_ds, batch_size=32, shuffle=True, num_workers=4)


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# hyperparams
num_epochs = 15

# train loop
outputs = []
for epoch in range(num_epochs):
    for image in data_loader_un:
        # size of batch torch.Size([B, 3, 128, 128]), 3 b/c RGB
        recon = model(image)
        loss = criterion(recon, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    LOGGER.info(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
    outputs.append((epoch, image, recon))


# Save the model and optimizer states
LOGGER.info("saving model")
save("autoencoder.pth", model, optimizer)

LOGGER.info("loading model")
# Load the model and optimizer states
model = AutoEncoder()
optimizer = optimizer.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
load("autoencoder.pth", model, optimizer)
