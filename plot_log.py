import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

letters = ["A","B","C","D","E","F"]

# Read data from file
for l in letters:

    log_name = l
    file_path = log_name + '.log'  # Update this with the correct file path
    df = pd.read_csv(file_path)

    # Replace 'NA' with NaN
    df.replace("NA", pd.NA, inplace=True)

    # Convert columns to appropriate dtypes
    df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')

    # Interpolate the NaN values in the val_loss column
    df['val_loss'] = df['val_loss'].interpolate()

    max_value = 0.0004  # Update this with your desired maximum value

    # Clip the loss values
    df['loss'] = df['loss'].clip(upper=max_value)
    df['val_loss'] = df['val_loss'].clip(upper=max_value)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linestyle='--')

    # Adding titles and labels
    plt.title("Val loss min: " + str(np.nanargmin(df["val_loss"].to_numpy())))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.legend()
    plt.grid(True)

    plt.savefig(log_name + ".png")

