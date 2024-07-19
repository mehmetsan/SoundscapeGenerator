from pathlib import Path

import pandas as pd

from file2img import convert_song

"""
    A script that combines the generated spectrograms and their 
    averaged valence-arousal scores in the same dataset
    Provide the paths to the DEAM dataset below here
"""

# Paths to the datasets
arousal = pd.read_csv('./deam_data/annotations/annotations averaged per song/dynamic/arousal.csv')
valence = pd.read_csv('./deam_data/annotations/annotations averaged per song/dynamic/valence.csv')
song_level_annotations_path = './deam_data/annotations/annotations averaged per song/song_level/'
averaged_2000 = pd.read_csv(f"{song_level_annotations_path}/static_annotations_averaged_songs_1_2000.csv")
averaged_2058 = pd.read_csv(f"{song_level_annotations_path}/static_annotations_averaged_songs_2000_2058.csv")

# Insert the path to the songs here
songs_path = './deam_data/MEMD_audio'

columns = ["song_id", "spectrogram_id", "valence", "arousal"]
df = pd.DataFrame(columns=columns)
df_data = []

mp3_files = sorted(Path(songs_path).glob('*.mp3'), key=lambda x: int(x.stem))
for song_index, song_path in enumerate(mp3_files):
    spectrograms_path = convert_song(file_path=song_path)

    song_id = Path(song_path).stem
    print(f"Finished converting song {song_id}")

    spectrograms = sorted(Path(spectrograms_path).glob('*.png'), key=lambda x: int(x.stem.split('_')[1]))

    for spec_index, spectrogram in enumerate(spectrograms):
        arousal_columns = [f"sample_{15000 + (spec_index * 5000) + (i * 500)}ms" for i in range(10)]
        valence_columns = [f"sample_{15000 + (spec_index * 5000) + (i * 500)}ms" for i in range(10)]

        related_arousal = arousal.loc[song_index, arousal_columns].tolist()
        related_valence = valence.loc[song_index, valence_columns].tolist()

        spectrogram_arousal = sum(related_arousal) / len(related_valence)
        spectrogram_valence = sum(related_valence) / len(related_valence)

        new_row = {
            "song_id": song_id,
            "spectrogram_id": spec_index,
            "arousal": spectrogram_arousal,
            "valence": spectrogram_valence
        }
        df_data.append(new_row)

df = pd.DataFrame(data=df_data, columns=columns)
df.to_csv('result.csv', index=False)

