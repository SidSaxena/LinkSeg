<div  align="center">
  
# LinkSeg: Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis

[Morgan Buisson](https://morgan76.github.io/)<sup>1</sup>,
[Brian McFee](https://brianmcfee.net/)<sup>2,3</sup>,
[Slim Essid](https://slimessid.github.io/research/)<sup>1</sup> <br>
<sup>1</sup>  LTCI, Télécom Paris, Institut Polytechnique de Paris, France <br> <sup>2</sup>  Music and Audio Research Laboratory, New York University, USA <br> <sup>3</sup> Center for Data Science, New York University, USA
  
<p align="center">
<img src="linkseg.png" width="500">
</p align="center">
</div>

This repository contains the official [PyTorch](http://pytorch.org/) implementation of the paper [Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis](https://hal.science/hal-04665063/) 
presented at ISMIR 2024. 

We introduce **LinkSeg**, a novel approach to music structure analysis based on pairwise link prediction. This method consists in predicting whether any pair of time instants within a track belongs to the same structural element (segment or section). This problem comes down to classifying each individual component (i.e. link) of the track's self-similarity matrix into one of the three categories: "same-segment", "same-section" or "different section". The link features calculated from this task are then combined with frame-wise features through a graph attention module to predict segment boundaries and musical section labels. 

This project is focused on the segmentation of popular music genres, therefore, predicted section labels follow a 7-class taxonomy containing: "Intro", "Verse", "Chorus", "Bridge", "Instrumental", "Outro" and "Silence". 

This repository provides code for training the system from scratch along with some checkpoints for predicting the structure of new tracks. 

## Table of Contents
0. [Requirements](#requirements)
0. [Dataset](#dataset)
0. [Training](#training)
0. [Inference](#inference)
0. [Apple Silicon Notes](#apple-silicon-notes)
0. [Citing](#citing)
0. [Contact](#contact)

## Requirements

### Option 1: Using uv (Recommended)

Install FFmpeg:

**Ubuntu:**
```shell
sudo apt install ffmpeg
```

**macOS:**
```shell
brew install ffmpeg
```

Install uv and set up the environment:
```shell
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/SidSaxena/LinkSeg.git
cd LinkSeg

# Install dependencies (madmom installed automatically from git)
uv sync

# Activate the environment
source .venv/bin/activate
```

### Option 2: Using conda

Install FFmpeg:

**Ubuntu:**
```shell
sudo apt install ffmpeg
```

**macOS:**
```shell
brew install ffmpeg
```

Create environment and install dependencies:
```shell
conda create -n YOUR_ENV_NAME python=3.11
conda activate YOUR_ENV_NAME
pip install git+https://github.com/CPJKU/madmom
pip install -r requirements.txt
```

## Dataset
The dataset structure follows that of the [MSAF](https://ismir2015.ismir.net/LBD/LBD30.pdf) package:
```
dataset/
├── audio                   # audio files (.mp3, .wav, .aiff)
├── audio_npy               # audio files (.npy)
├── features                # feature files (.json)
└── references              # references files (.jams) (if available, necessary for training)
```

**Important:** The preprocessing and inference scripts expect audio files to be located directly in the `audio/` subdirectory. If your audio files are organized in nested subdirectories (e.g., `dataset/audio/artist/track.mp3`), you have two options:

1. **Flatten the structure** (recommended): Create symlinks or copy all audio files into a single `audio/` directory:
   ```shell
   mkdir -p dataset/audio
   find /path/to/your/music -name "*.mp3" -exec ln -s {} dataset/audio/ \;
   ```

2. **Use a flat directory**: Place all audio files directly in `dataset/audio/` without subdirectories.

To preprocess some dataset, run:
```shell
python preprocess_data.py --data_path <dataset_path>
```
This will handle the creation of [`dataset/audio_npy/`](dataset/audio_npy/) (conversion to numpy files) and [`dataset/features/`](dataset/features/) files (beat estimation). 

## Training
To train a new LinkSeg model, run:
```shell
python train.py --data_path <dataset_path>
```

The default label taxonomy contains 7 section labels: "Intro", "Verse", "Chorus", "Bridge", "Instrumental", "Outro" and "Silence". A second taxonomy containing "Pre-chorus" and "Post-chorus" labels can be used:
```shell
python train.py --data_path <dataset_path> --nb_section_labels 9
``` 

## Inference
To make predictions using a trained model, first make sure that the test dataset is processed: 
```shell
python preprocess_data.py --data_path <test_dataset_path>
```

Then run:
```shell
python predict.py --test_data_path <test_dataset_path> --model_name <path_to_model>
```

To use the 7-class pre-trained model (on a 75% split of Harmonix), run:
```shell
python predict.py --test_data_path <dataset_path> --model_name ../data/model_7_classes.pt
```

To use the 9-class pre-trained model, run:
```shell
python predict.py --test_data_path <dataset_path> --model_name ../data/model_9_classes.pt
```

By default, segmentation predictions will be saved in [JAMS](https://jams.readthedocs.io/en/stable/quickstart.html) format under the [`dataset/predictions/`](dataset/predictions/) directory. 

Keep in mind that boundary predictions are calculated from the features of two consecutive time frames $x\prime \prime_{i}$, $x\prime \prime_{i+1}$ and the features $e\prime_{i,i+1}$ of the link connecting them. Therefore, boundary predictions fall **between** consecutive estimated beat locations. 

## Apple Silicon Notes

- **MPS (Metal Performance Shaders) Support**: The model automatically detects and uses MPS when available on Apple Silicon Macs.
- **FFT Operations**: Some operations (FFT/spectrogram) are not yet supported on MPS and will automatically fall back to CPU. This is handled transparently.
- **Performance**: For best performance on Apple Silicon, the model will use CPU mode with MPS fallback enabled.
- **Environment Variable**: The script automatically sets `PYTORCH_ENABLE_MPS_FALLBACK=1` to handle unsupported operations.

## Segmentation Example

<div  align="center">
<p align="center">
<img src="segmentation_example_midnight_city.png" width="1000">
</p align="center">
</div>

Segmentation results for the track ***M83 - Midnight City*** from the Harmonix dataset. The top two rows display the class activation and boundary curves over time. The bottom rows show the final predictions and the reference annotations. Black and red dotted lines indicate predicted and annotated segment boundaries.

<div  align="center">
<p align="center">
<img src="ssm_example_midnight_city.png" width="800">
</p align="center">
</div>

The self-similarity matrix of the track ***M83 - Midnight City*** obtained from the output embeddings of the graph attention module. Red dashed lines indicate annotated (left) and predicted (right) segment boundaries.

## Citing
```bib
@inproceedings{buisson:hal-04665063,
  title={Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis},
  author={Buisson, Morgan and McFee, Brian and Essid, Slim},
  booktitle={Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```

## Contact
morgan.buisson76@gmail.com
