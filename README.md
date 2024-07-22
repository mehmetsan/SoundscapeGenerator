# SoundscapeGenerator
A project to use Stable Diffusion to generate soundscapes with emotion tags as input

# Dependencies
- For setup you need to create a conda environment, if you don't want the whole conda package you can get a minified version from here:
  - https://docs.anaconda.com/miniconda/miniconda-install/
- Create a conda virtual environment
  - `conda create -n myenv python=3.9`
- Activate/Deactivate the environment
  - `activate myenv`
  - `deactivate myenv`
- Add the following channels to the conda to look for the packages
  - `conda config --env --add channels conda-forge`
  - `conda config --env --add channels anaconda`
- Install the dependencies:
  - `conda install --yes --file requirements.txt`
- Visit https://pytorch.org/ and formulate your conda command based on your OS, such as:
  - OSX: `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
  - Windows (check your CUDA version): `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
# Setup
- To begin with activate your env and run the script preparation.py to download the required DEAM dataset.
  - `python preparation.py`
- Create the spectrograms
  - `python create_spectrograms.py`
- Prepare the dataset
  - `python prepare_dataset.py`
- Categorize the spectrograms based on their emotion classes
  - `python categorize_spectrograms.py`
- (Optional) If you want to generate K-means clustering map
  - `python cluster_emotions.py`
## References

- Used the helper functions from, great help:
  - https://github.com/chavinlo/riffusion-manipulation.git