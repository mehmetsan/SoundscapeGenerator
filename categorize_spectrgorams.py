import os
import shutil
from pathlib import Path

import pandas as pd

if not os.path.exists('./spectrograms'):
    print('Spectrograms are missing, run the create_spectrograms.py script first to generate the spectrograms')
else:
    if not os.path.exists('./result.csv'):
        print("Result dataset is missing, run the prepare_dataset.py script first")

    else:
        data = pd.read_csv('./result.csv')

        # Clean the directory if it exists
        if os.path.exists('./categorized_spectrograms'):
            shutil.rmtree('./categorized_spectrograms')

        # For each spectrogram, copy the image to the assigned emotion class directory
        for index, row in data.iterrows():
            emotion = row['emotion']
            song_id = row['song_id']
            spectrogram_id = row['spectrogram_id']

            target_path = f"./categorized_spectrograms/{emotion}"
            os.makedirs(target_path, exist_ok=True)

            original_file_path = Path(f"./spectrograms/{song_id}/{song_id}_{spectrogram_id}.png")

            shutil.copy(original_file_path, target_path)

        print('Finished categorizing the spectrograms, you can find them inside ./categorized_spectrograms directory')