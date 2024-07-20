import os
from pathlib import Path

import pandas as pd

"""
    A script that combines the generated spectrograms and their 
    averaged valence-arousal scores in the same dataset
"""

song_level_annotations_path = 'deam_data/annotations/annotations averaged per song/song_level/'
paths = {
    'arousal': './deam_data/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv',
    'valence': './deam_data/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv',
    'averaged_2000': f"{song_level_annotations_path}/static_annotations_averaged_songs_1_2000.csv",
    'averaged_2058': f"{song_level_annotations_path}/static_annotations_averaged_songs_2000_2058.csv",
}

arousal = pd.read_csv(paths['arousal'])
valence = pd.read_csv(paths['valence'])

# Dataframes
averaged_2000 = pd.read_csv(paths['averaged_2000'])
averaged_2058 = pd.read_csv(paths['averaged_2058'])
averaged_combined = pd.concat([averaged_2000, averaged_2058])

averaged_combined.columns = averaged_combined.columns.str.replace(' ', '')

columns = ['song_id', 'spectrogram_id', 'valence', 'arousal', 'song_arousal_std', 'song_valence_std']
df = pd.DataFrame(columns=columns)
df_data = []

if not os.path.exists('./spectrograms'):
    print('Spectrograms data missing, use the create_spectrograms.py script first to generate the spectrograms')

else:
    sorted_folders = sorted(
        (folder for folder in Path('./spectrograms').iterdir() if folder.is_dir()),
        key=lambda x: int(x.stem)
    )

    for song_index, song_folder in enumerate(sorted_folders):

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
                    'song_arousal_std': averaged_combined.iloc[song_index]['arousal_std'],
                    'song_valence_std': averaged_combined.iloc[song_index]['valence_std']
                }
                df_data.append(new_row)


df = pd.DataFrame(data=df_data, columns=columns)
df.to_csv('result.csv', index=False)

