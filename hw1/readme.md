# DeepMIR 2023fall Homework 1

Homework 1 for DeepMIR 2023fall.

## Getting Started

These instructions will guide you on how to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Ensure you have [Anaconda](https://www.anaconda.com/products/individual) installed on your machine to manage dependencies and virtual environments.

### Setting Up the Environment

1. Create a new Conda environment named `deepmir` with Python 3.10:
    ```bash
    conda create -n deepmir python=3.10
    ```

2. Activate the `deepmir` environment:
    ```bash
    conda activate deepmir
    ```

3. Install the necessary dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

1. Perform source separation using Demucs:
    ```bash
    python ss_preprocess.py  # source separation using Demucs
    ```
    Please ensure that the `artist20/test/*.mp3` and `artist20/test/*wav` folder/files are in the same directory as `ss_preprocess.py`.
    The separated sources will be saved in `artist20/test_separated/{song_ids}/mdx_extra/{song_ids}/vocals.mp3`.
    
    The directory structure should look like this:

    ```
    📂 root
    ├── 📂 artist20
        ├── 📂 test
        │   ├── 🎵 *.mp3
        │   └── 🎵 *.wav
        └── 📂 test_separated
            └── 📂 {song_ids}
                └── 📂 mdx_extra
                    └── 📂 {song_ids}
                        ├── 🎵 vocals.mp3
                        └── 🎵 no_vocals.mp3
    ```


    


2. Test the model:
    ```bash
    python test.py 
    ```

## Authors

- Yueh-Po Peng (yuuehpo.peng@gmail.com)