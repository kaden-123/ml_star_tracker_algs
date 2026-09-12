# ML-Based Star Tracker

An ML-based star-tracker pipeline that estimates camera attitude from a simulated camera orientation. The project generates visible star centroids and pixel projections, converts each candidate star field into a radial histogram, and uses a neural network to predict star IDs. Three identified stars are then passed to Davenport's q-method to estimate the attitude quaternion and recover the camera orientation.

The neural-network star-identification approach is based on:

> D. Rijlaarsdam, H. Yous, J. Byrne, D. Oddenino, G. Furano, and D. Moloney, “Efficient Star Identification Using a Neural Network,” *Sensors*, vol. 20, no. 13, article 3684, 2020. https://doi.org/10.3390/s20133684

## Pipeline

1. Load stars from the HYG catalog.
2. Point a simulated camera toward a selected or random orientation.
3. Project visible stars into image-plane centroid coordinates.
4. Generate radial-histogram features for neural-network training or inference.
5. Predict three star IDs with the trained network.
6. Use the identified camera and catalog vectors in Davenport's q-method.
7. Return the estimated attitude quaternion, right ascension, declination, and angular error.

## Main files

- `camera.py` — camera orientation and centroid projection simulation.
- `attitude.py` — quaternion, Euler-angle, and direction-cosine-matrix utilities.
- `algorithm.py` — Davenport's q-method attitude estimator.
- `pole_nn/pole_nn.py` — PyTorch model and HDF5 dataset loader.
- `pole_nn/pole_nn_data_gen.py` — simulated training-data generator.
- `pole_nn/pole_nn_training.py` — model training, testing, and hyperparameter tuning.
- `model_evaluation.py` — neural-network inference for candidate stars.
- `test_pipeline.py` — end-to-end star-identification and attitude-estimation pipeline.

## Setup

Install the Python dependencies:

```bash
pip install requirements.txt
```

Download `hygdata_v42.csv` from the HYG star database and place it in the project's `data/` directory.
https://www.astronexus.com/projects/hyg


## Generate training data


```bash
python -m star_tracker.pole_nn.pole_nn_data_gen --help
```

## Hypertune the model

```bash
python -m star_tracker.pole_nn.pole_nn_training hypertune --help  
```

Use `--device cuda` when CUDA is available.

## Train the model

```bash
python -m star_tracker.pole_nn.pole_nn_training train --help  
```

Use `--device cuda` when CUDA is available.

## Test saved weights

```bash
python -m star_tracker.pole_nn.pole_nn_training --device cpu test --file-name model_weights.pth
```

The bin count and hidden-layer sizes must match the configuration used to train the saved model.

## Run the full pipeline

Point the camera at an HR catalog star with a specified roll:

```bash
python -m star_tracker.test_pipeline --model pole_nn --id_point <HR_ID> <ROLL_DEGREES>
```

Use a random camera orientation:

```bash
python -m star_tracker.test_pipeline --model pole_nn --rand_point
```

Or provide right ascension in hours, declination in degrees, and roll in degrees:

```bash
python -m star_tracker.test_pipeline --model pole_nn --celest_point <RA_HOURS> <DEC_DEGREES> <ROLL_DEGREES>
```
Only current ML model is pole_nn,

## Data and model files

Generated HDF5 datasets, trained model weights, experiment outputs, and the downloaded star catalog should remain in `data/`
