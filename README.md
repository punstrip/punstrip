# desyl

This repository contains source code for the DEbug SYmbol Learning (DESYL) project, formerly known as punstrip. 

The project's aim is to combine deep learning methods and binary analysis to name symbols in stripped binaries.

The github repository is a mirror of the true repository and may have the latest research held back.

Most of the functional code is under `src/classes/`. Some documentation is written in `docs/`. 

To get started copy `example.conf` to `desyl.conf` in the repository root and configure the project.

![unstrip](res/unstrip.gif)


### Dependencies

- Graph2Vec https://github.com/benedekrozemberczki/graph2vec.git
- See `requirements.txt`

## Testing

```desyl-db:~/desyl/src# python3 -m unittest tests.crf.TestCRF.test_knowns_iter```
