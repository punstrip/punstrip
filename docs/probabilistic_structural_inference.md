# Overview

    - Complete everything in `./docs/probabilistic_fingerprint.md`
    - Generate a set of new PSI weights and learn pareters


## Training

Run `./scripts/train_crf.py`

## Testing

We currently assume function boundaries are known. You may test using a 3rd party tool to supply this information.
Potential tools include (in order of assumed effectiveness):

    - JIMA
    - Nucleus
    - IDA Pro
    - Ghidra
    - Binary Ninja
    - Radare 2

To infer function names on an unknown binary run `python3 ./scripts/infer_crf.py /unknown_binary`
