"""
Main Neural Pipeline Script
---------------------------
This script loades neural data, analog event data, and audio data from specific streams
and specified electrodes, and saves the data into pickle files.
"""

# ========================
# Imports and Setup
# ========================

import os
import sys
import pickle
import importlib
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Local imports (make sure your pipeline path is accessible)
sys.path.append(
    os.path.abspath(
        r"C:\Users\denhal.WISMAIN\Weizmann Institute Dropbox\DeanAdam Halperin\BrainValue\Code\neural_pipeline"
    )
)

# Import and reload local modules (useful for development)
import ripple_io
import ripple_io.ripple_loader
importlib.reload(ripple_io.ripple_loader)

from ripple_io.ripple_loader import RippleLoader
from processing.preprocessing import DataPreProcessing  # Optional, currently unused


# ========================
# User Input (Folder Dialog)
# ========================

# Hide Tkinter root window (we only want the folder dialog)
root = tk.Tk()
root.withdraw()

# Ask user to select a data folder
file_path = filedialog.askdirectory(title="Select a folder")

# ========================
# Configuration Parameters
# ========================

base_file_name = 'datafile0027' # default is usually 'datafile' - but can change based on recording setup

channels_in_zero_index = True  # True = Match online indexing (starting from 0)
streams = ['spikes', 'raw', 'lfp', 'hi-res'] # List of streams to load

# Define electrode settings
dead_electrodes = (
    [130, 131, 260, 268, 264, 273] +
    list(range(274, 289)) +
    list(range(64, 128))
)
wanted_electrodes = 'all'  # Can also be a list, e.g., [263]

# Define segment range to load
first_segment = 0
last_segment = 999  # Use 999 to load all available segments


# ========================
# Main Processing Loop
# ========================

for stream in streams:
    print(f"\n--- Loading stream: {stream} ---")

    if stream == 'spikes':
        # in order to load analog event channels, and analog audio channels only once, we load it here
        analog_channels = [1, 2]  # e.g., audio or behavioral analog channels
        load_audio = True
    else:
        analog_channels = None
        load_audio = False

    # Initialize data loader
    loader = RippleLoader(
        wanted_streams=[stream],
        proc_single=False,
        wanted_electrodes=wanted_electrodes,
        event_port=None,
        analog_channels=analog_channels,
        spike_units=None,
        timestamp_units_in_seconds=1,
        first_segment=first_segment,
        last_segment=last_segment,
        dead_electrode=dead_electrodes,
        load_audio=load_audio,
        channels_in_zero_index=channels_in_zero_index
    )

    # Load data (returns multiple data types)
    neural_data, analog_event_data, _, audio_data = loader.load(
        file_path, base_file_name=base_file_name
    )

    # ========================
    # Save Processed Data
    # ========================

    # Create output folder if it doesn't exist
    output_folder = os.path.join(file_path, "processed_data")
    os.makedirs(output_folder, exist_ok=True)

    # Save neural data
    neural_filename = f'neural_data_{stream}.pkl'
    with open(os.path.join(output_folder, neural_filename), 'wb') as f:
        pickle.dump(neural_data, f)
    print(f"Saved neural data → {neural_filename}")

    # Save event data (if available)
    if analog_event_data:
        with open(os.path.join(output_folder, 'event_data.pkl'), 'wb') as f:
            pickle.dump(analog_event_data, f)
        print("Saved event data → event_data.pkl")

    # Save audio data (if available)
    if audio_data:
        with open(os.path.join(output_folder, 'audio_data.pkl'), 'wb') as f:
            pickle.dump(audio_data, f)
        print("Saved audio data → audio_data.pkl")
