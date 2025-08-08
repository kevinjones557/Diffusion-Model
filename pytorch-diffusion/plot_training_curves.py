import matplotlib.pyplot as plt

def parse_log_file(file_path):
    training_losses = []
    validation_losses = []

    current_epoch_losses = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Epoch [201/500]' in line:
                break
            
            # Check for validation loss
            if line.startswith("Epoch:") and "Validation Loss" in line:
                # Extract validation loss
                val_loss = float(line.split("Validation Loss:")[1].strip())
                validation_losses.append(val_loss)
                
                # Compute average training loss for the previous epoch
                if current_epoch_losses:
                    avg_training_loss = sum(current_epoch_losses) / len(current_epoch_losses)
                    training_losses.append(avg_training_loss)
                    current_epoch_losses = []  # Reset for the next epoch
            
            # Check for batch training loss
            elif "Loss:" in line and "Batch" in line:
                # Extract batch loss
                loss_part = line.split("Loss:")[1]
                batch_loss = float(loss_part.split(",")[0].strip())
                current_epoch_losses.append(batch_loss)
        
        # Handle the last epoch's training loss
        if current_epoch_losses:
            avg_training_loss = sum(current_epoch_losses) / len(current_epoch_losses)
            training_losses.append(avg_training_loss)

    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses):
    epochs = range(1, len(training_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses[1:], label="Training Loss", marker='o')
    plt.plot(epochs, validation_losses[1:], label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

# File path to the log file
file_path = "log_2000_float16_precision_no_scaler.txt"  # Replace with your actual file path

# Parse log file and extract losses
training_losses, validation_losses = parse_log_file(file_path)

# Plot the losses
plot_losses(training_losses, validation_losses)