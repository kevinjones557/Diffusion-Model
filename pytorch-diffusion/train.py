from model import *
from data import get_dataloader

import torch
import torch.optim as optim
import time
import torch.cuda.amp as amp

# Initialize the model
t_range = 200  # Number of steps
image_size = (1, 3, 32, 32)
img_depth = 3  # Number of channels in the image
dataset_choice = "Cifar-10"
batch_size=256
device = 'cuda'

# scaler = amp.GradScaler()

train_loader = get_dataloader(dataset_name=dataset_choice, batch_size=batch_size)
validation_loader = get_dataloader(dataset_name=dataset_choice, batch_size=batch_size, split='validation')

model = DiffusionModel(in_size=32 * 32, t_range=t_range, img_depth=img_depth, device=device).to(device)
# model.load_state_dict(torch.load("model_2000_bfloat16_precision.pth"))
optimizer = optim.Adam(model.parameters(), lr=2e-4)
num_epochs = 500
epoch_start = 0

def compute_validation_loss(model, validation_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for batch_idx, (batch, _) in enumerate(validation_loader):
            # Move data to device
            batch = batch.to(device)

            # Compute loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = model.get_loss(batch, batch_idx)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def lower_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr

def print_and_write(file, string):
    print(string)
    file.write(string + '\n')
    file.flush()

# Training loop
log_file = open(f"log_{t_range}_bfloat16_precision.txt", "a")
for epoch in range(epoch_start, num_epochs):
    if epoch == num_epochs * 0.8:
        lower_lr(optimizer, 2e-5)
    for batch_idx, (batch, _) in enumerate(train_loader):
        model.train()
        t1 = time.time()
        batch = batch.to(device)

        # Compute loss
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model.get_loss(batch, batch_idx)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)  
        # scaler.update()

        t2 = time.time()
        dt = (t2 - t1) * 1000
        ms_per_img =  dt / batch_size
        # Print loss every 10 batches
        if batch_idx % 10 == 0:
            s = f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}, Time / Image: {ms_per_img:.4f}"
            print_and_write(log_file, s)
    val_loss = compute_validation_loss(model, validation_loader, device)
    s = f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}"
    print_and_write(log_file, s)
    torch.save(model.state_dict(), f"model_{t_range}_bfloat16_precision.pth")
