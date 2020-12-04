# Overview

To train a new model one needs to adhear to the following procedure:

    - Install the requirements/set up databases
    - Analyse the executable binaries
    - Set model hyperparameters and generate symbol vectors
    - Train a model

## Analysis

After setting up desyl, ensure the configuration file `desyl.conf` is present in the projects root directory and is correct.

Run `./src/scripts/add_binaries.py` and `./src/scripts/add_libraries.py` scripts with the command line argument being the directory containing the binaries to be analysed.

NB: `find /elf_binaries_dir | parallel -P 128 python3.8 ./scripts/add_binaries.py`

## Defining a new experiment

Define a new experiment in your `desyl.conf` file.

Generate and save the experiment settings by running `./src/classes/experiment.py`

Generate symbol vectors for the current experiment settings by running `./src/scripts/generate_symbol_embeddings.py`

## Training a model

Run `./src/scripts/train_fingerprint.py`.



