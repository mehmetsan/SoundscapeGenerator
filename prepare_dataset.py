import os
from pathlib import Path

import pandas as pd
import numpy as np

"""
    A script that combines the generated spectrograms and their 
    averaged valence-arousal scores in the same dataset
"""


def assign_emotions(valence_score, arousal_score):
    # Calculate the angle in radians between the positive x-axis and the point (x, y)
    angle = np.arctan2(arousal_score, valence_score)

    # Normalize the angle to be between 0 and 2*pi
    if angle < 0:
        angle += 2 * np.pi

    # First Quadrant: 0 to pi/2
    if 0 <= angle <= np.pi / 6:
        return 'happy'
    elif np.pi / 6 < angle <= np.pi / 3:
        return 'delighted'
    elif np.pi / 3 < angle <= np.pi / 2:
        return 'excited'

    # Second Quadrant: pi/2 to pi
    elif np.pi / 2 < angle <= 2 * np.pi / 3:
        return 'tense'
    elif 2 * np.pi / 3 < angle <= 5 * np.pi / 6:
        return 'angry'
    elif 5 * np.pi / 6 < angle <= np.pi:
        return 'frustrated'

    # Third Quadrant: pi to 3pi/2
    elif np.pi < angle <= 7 * np.pi / 6:
        return 'depressed'
    elif 7 * np.pi / 6 < angle <= 4 * np.pi / 3:
        return 'bored'
    elif 4 * np.pi / 3 < angle <= 3 * np.pi / 2:
        return 'tired'

    # Fourth Quadrant: 3pi/2 to 2pi
    elif 3 * np.pi / 2 < angle <= 5 * np.pi / 3:
        return 'calm'
    elif 5 * np.pi / 3 < angle <= 11 * np.pi / 6:
        return 'relaxed'
    elif 11 * np.pi / 6 < angle <= 2 * np.pi:
        return 'content'


song_level_annotations_path = './deam_data/annotations/annotations averaged per song/song_level/'
paths = {
    'arousal': './deam_data/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv',
    'valence': './deam_data/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv',
    'averaged_2000': f"{song_level_annotations_path}/static_annotations_averaged_songs_1_2000.csv",
    'result': './result.csv'
}

arousal = pd.read_csv(paths['arousal'])
valence = pd.read_csv(paths['valence'])

# Dataframes
averaged_2000 = pd.read_csv(paths['averaged_2000'])
averaged_2000.columns = averaged_2000.columns.str.replace(' ', '')

columns = ['song_id', 'spectrogram_id', 'valence', 'arousal', 'song_arousal_std', 'song_valence_std', 'emotion']
df = pd.DataFrame(columns=columns)
df_data = []

if not os.path.exists('./spectrograms'):
    print('Spectrograms data missing, use the create_spectrograms.py script first to generate the spectrograms')

else:
    if os.path.exists(paths['result']):
        os.remove(paths['result'])

    sorted_spectrogram_folders = sorted(
        (folder for folder in Path('./spectrograms').iterdir() if folder.is_dir()),
        key=lambda x: int(x.stem)
    )

    for song_index, song_folder in enumerate(sorted_spectrogram_folders):

        if song_folder.is_dir():
            song_id = song_folder.stem
            song_spectrograms = sorted(Path(song_folder).glob('*.png'), key=lambda x: int(x.stem.split('_')[1]))

            for spec_index, spectrogram in enumerate(song_spectrograms):
                arousal_columns = [f"sample_{15000 + (spec_index * 5000) + (i * 500)}ms" for i in range(10)]
                valence_columns = [f"sample_{15000 + (spec_index * 5000) + (i * 500)}ms" for i in range(10)]

                related_arousal = arousal.loc[song_index, arousal_columns].tolist()
                related_valence = valence.loc[song_index, valence_columns].tolist()

                spectrogram_arousal = sum(related_arousal) / len(related_valence)
                spectrogram_valence = sum(related_valence) / len(related_valence)

                new_row = {
                    'song_id': song_id,
                    'spectrogram_id': spec_index,
                    'arousal': spectrogram_arousal,
                    'valence': spectrogram_valence,
                    'song_arousal_std': averaged_2000.iloc[song_index]['arousal_std'],
                    'song_valence_std': averaged_2000.iloc[song_index]['valence_std'],
                    'emotion': assign_emotions(valence_score=spectrogram_valence, arousal_score=spectrogram_arousal)
                }
                df_data.append(new_row)

# Save the dataset
df = pd.DataFrame(data=df_data, columns=columns)
df.to_csv('result.csv', index=False)
print('Dataset created successfully')
