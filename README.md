# SoundscapeGenerator
A project to use Stable Diffusion to generate soundscapes with emotion tags as input

# Dependencies
- For setup you need to create a conda environment, if you don't want the whole conda package you can get a minified version from here:
  - https://docs.anaconda.com/miniconda/miniconda-install/
- Create a conda virtual environment
  - `conda create -n myenv python=3.9`
- Activate the environment
  - `conda activate myenv`
- Deactivate the environment
  - `conda deactivate myenv`
- Install the dependencies:
  - `conda install --yes --file requirements.txt`
- Visit https://pytorch.org/ and formulate your conda command based on your OS, such as:
  - `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
- Install open-cv separately
  - `conda install opencv`

# Setup
- To begin with activate your env and run the script preparation.py to download the required DEAM dataset.
  - `python3 preparation.py`
- Create the spectrograms
  - `python3 create_spectrograms.py`
- Prepare the dataset
  - `python3 prepare_dataset.py`
- Categorize the spectrograms based on their emotion classes
  - `python3 categorize_spectrograms.py`
- (Optional) If you want to generate K-means clustering map
  - `python3 cluster_emotions.py`
## References

- Used the helper functions from, great help:
  - https://github.com/chavinlo/riffusion-manipulation.git