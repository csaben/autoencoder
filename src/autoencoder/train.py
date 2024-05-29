import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from autoencoder.ae import AutoEncoder
from autoencoder.transforms import get_dataset

load_dotenv("data.env")

infected_ds = get_dataset(os.getenv("infected_patients_directories"))
uninfected_ds = get_dataset(os.getenv("uninfected_patients_directories"))
root_dir = os.getenv("infected_patients_directories")
data_loader_in = DataLoader(infected_ds, batch_size=32, shuffle=True, num_workers=4)
data_loader_un = DataLoader(uninfected_ds, batch_size=32, shuffle=True, num_workers=4)


print(data_loader_in, data_loader_un)


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# hyperparams
num_epochs = 2

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

    print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
    outputs.append((epoch, image, recon))
