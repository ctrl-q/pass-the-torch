# PassTheTorch
Framework for reproducible and trackable Machine-Learning experiments.
This project's aim is to allow independent coding of ML models, data loaders, progress tracking and hyperparameter optimization

## Setup
```sh
git clone https://ctrl-q/pass-the-torch
cd pass-the-torch
pip install -r requirements.txt
```
## Models
* The base classes for all models are defined in [models/base.py](models/base.py)
* The other files in that folder are provided as examples


## Experiments
All possible experiments are stored in the [experiments](experiments) folder. <br>
Some hyperparameters are common to all experiments and some are particular to an experiment <br><br>
All hyperparameters except for `datapath` and `trials` can be specified as:
1. a value
1. a tuple of 2 values, which will be interpreted as a range*
1. a list of multiple values, which will be interpreted as a discrete list of choices*

All hyperparams that are lists or tuples will be tuned via [scikit-optimize](https://scikit-optimize.github.io/) for *`trials`* iterations

\* Please quote tuples or lists in the command line. *e.g.* --lr (0.001, 0.1) -> --lr '(0.001, 0.1)' <br>

Training progress will be available via [tensorboardX](https://tensorboardx.readthedocs.io/)


## Utils
Can be used to store any code for preparing your data for training, _e.g._ for dataloaders, logging, or anything else you could think of.

## Configuration
* Add your own model to the [models](models) folder. The model should subclass `PyTorchModel` or `SKLearnModel`

## Running
1. Choose an experiment from the [experiments](experiments) folder
1. Run `"python3 -m experiments.<experiment_name> -h"` to get the list of hyperparameters*
1. Run `"python3 -m experiments.<experiment_name>"` with hyperparams specified**

The experiments will be saved in the following path: `<experiment name>/<hyphen-separated hyperparams in the same order as in the argparse>`

\* All experiments **must** be run from this folder, and **not** from the [experiments](experiments) folder <br>
\*\* The double quotes are needed
