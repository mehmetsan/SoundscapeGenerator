import os

from pathlib import Path

from file2img import convert_song

"""
    A script to generate the spectrogram files using the 
    downloaded mp3 files from the DEAM Dataset 
"""

songs_path = './deam_data/MEMD_audio'

if not os.path.exists(songs_path):
    print('Songs data missing, use the preparation.py script first to get the data')
else:
    # Get the mp3 files and iterate them
    mp3_files = sorted(Path(songs_path).glob('*.mp3'), key=lambda x: int(x.stem))
    for song_index, song_path in enumerate(mp3_files):
        song_name = Path(song_path).stem

        spectrograms_path = convert_song(song_path=song_path)

        if not spectrograms_path:
            print(f"Song {song_name} was already converted, skipped...")
        else:
            print(f"Finished converting song {song_name}")
