# Automatic recognition of epileptic seizures in the EEG

Repository to demonstrate the packaging process for algorithms to be evaluated with [SzCORE](https://github.com/esl-epfl/szcore/).

This repository re-implements [SzCORE compatible reproduction of Gomez, 'Automatic seizure detection based on imaged-EEG signals through fully convolutional networks', Gomez. Scientific reports, 2020.](https://doi.org/10.1016/0013-4694(82)90038-4).

The python code along with dependencies are contained in the [`fcn_2020`](fcn_2020/) folder. The code is developed to run automated seizure detection on BIDS / SzCORE standardized `EDF` files. The code is expected to output a HED-SCORE / SzCORE `TSV` annotation file.

The python code is then packaged in a docker container. The container contains two volumes (`/data`, `/output`), used respectively, to hold the input `EDF` and output `TSV.` Upon entry the container should run the algorithm. It expects the input filename as an environment variable `$INPUT` and the output filename as an environment variable `$OUTPUT`.

