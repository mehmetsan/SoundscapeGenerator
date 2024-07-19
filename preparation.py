import os
import wget
import zipfile

# URLS for the DEAM dataset
base_url = 'http://cvml.unige.ch/databases/DEAM'
audio_url = f"{base_url}/DEAM_audio.zip"
annotations_url = f"{base_url}/DEAM_Annotations.zip"

url_file_namings = {
    audio_url: 'MEMD_audio',
    annotations_url: 'annotations'
}


def download_and_extract(url, download_directory):
    path_suffix = url_file_namings[url]
    path = f"./deam_data/{path_suffix}"
    if os.path.exists(path):
        print(f"{url} is already downloaded, skipping...")
        return

    print(f"Downloading.. {url}")
    # Download the zip file
    wget.download(url, out=f"{download_directory}/file.zip")
    print(f"Finished")

    # Extract the contents
    with zipfile.ZipFile(f"{download_directory}/file.zip", 'r') as zip_ref:
        zip_ref.extractall(download_directory)

    # Delete the zip file
    os.remove(f"{download_directory}/file.zip")


os.makedirs('deam_data', exist_ok=True)
target_directory = 'deam_data'

download_and_extract(audio_url, target_directory)
download_and_extract(annotations_url, target_directory)
