# CaloPointFlow II

## Abstract

In particle physics experiments, simulating calorimeter showers is computationally demanding, affecting the efficiency and accuracy of these experiments. Generative Machine Learning (ML) models have improved and sped up traditional physics simulation processes, but their use has largely been limited to fixed detector readout geometries. CaloPointFlow introduced a pioneering model capable of generating a calorimeter shower as a point cloud. This study advances the field with CaloPointFlow II, featuring several significant enhancements over its predecessor. These improvements include a novel dequantization technique, named CDF-Dequantization, and a new normalizing flow architecture, dubbed DeepSet-Flow. The updated model was tested using the fast Calorimeter Simulation Challenge (CaloChallenge) Dataset II and III, demonstrating its effectiveness and potential applications in particle physics simulations.

## Key Features

- **CDF-Dequantization:** A novel technique that improves the quality of generated point clouds by effectively handling the quantization of data, leading to more accurate simulations.
- **DeepSet-Flow Architecture:** A state-of-the-art normalizing flow architecture that efficiently processes set-structured data, enhancing the model's ability to generate complex calorimeter showers.
- **Enhanced Efficiency:** Significant improvements in computational efficiency, enabling faster simulations without sacrificing accuracy.
- **Versatile Application:** Tested with the CaloChallenge Dataset II and III, showcasing its applicability to a wide range of particle physics experiments.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or later
- PyTorch 1.8 or later
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/simonschnake/CaloPointFlow.git
cd CaloPointFlow
```

2. Install the required Python packages, we use `python 3.10.5` and `pip` :

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2%2Bcu121.html
pip install click==8.1.7 zuko==1.1.0 tqdm==4.66.4 omegaconf==2.3.0 numba-0.59.1 pytorch-lightning==2.3.2 matplotlib==3.9.1 h5py==3.11.0 scipy==1.14.0 mplhep==0.3.50 tensorboardX==2.6.2.2
pip install .
```

### Usage

To train the model use
```bash
calopointflow train --help

Usage: calopointflow train [OPTIONS] [KWARGS]...

Options:
  -m, --model TEXT       Model to use. Options: I, dsf, dsf_cdeq, II
                         [required]
  -d, --dataset INTEGER  Dataset to use. Options: 2, 3  [required]
  -ld, --log_dir TEXT    Path to save the logs  [required]
  --help                 Show this message and exit.
```

To generate new data use

```bash
calopointflow generate --help

Usage: calopointflow generate [OPTIONS] [KWARGS]...

Options:
  -m, --model TEXT       Model to use. Options: I, dsf, dsf_cdeq, II
                         [required]
  -d, --dataset INTEGER  Dataset to use. Options: 2, 3  [required]
  --ckpt_path TEXT       Path to the model checkpoint file  [required]
  --save_path TEXT       Path to save the generated data  [required]
  --help                 Show this message and exit.
```

To plot the histograms

```bash
 calopointflow plot --help

Usage: calopointflow plot [OPTIONS]

Options:
  -d, --dataset INTEGER           Dataset to use. Options: 2, 3  [required]
  -p, --plot TEXT                 What to plot.
                                  
                                  Options:
                                  
                                    all: Plot all available plots
                                  
                                    marginals: Plot the marginal distributions
                                    of the data in the z, alpha, and r
                                    dimensions
                                  
                                    layer_energies: Plot the energy
                                    distributions of the data in individual
                                    layer areas
                                  
                                    corrcoeff: Plot the correlation
                                    coefficients between the data in the z,
                                    alpha, and r dimensions
                                  
                                    cov_eigenvalues: Plot the histograms of
                                    eigenvalues of the covariance matrices of
                                    the individual showers
                                  
                                    means: Plot the shower means in the z,
                                    alpha, and r dimensions
                                  
                                    cell_energies: Plot the energy
                                    distributions of the data in individual
                                    cells
                                  
                                    num_hits: Plot the histogram of number of
                                    hits
  --save_path TEXT                Path to save the plots
  -g4, --geant4_data TEXT         Path to the Geant4 data  [required]
  -cpf, --calopointflow_data TEXT
                                  Path to the CaloPointFlow data  [required]
  --help                          Show this message and exit.
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- Fast Calorimeter Simulation Challenge (CaloChallenge) for providing the datasets.
