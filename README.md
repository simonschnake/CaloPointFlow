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

2. Install the required Python packages:

```bash
pip install .
```

### Usage

To start using CaloPointFlow II, follow these steps:

1. Prepare your dataset according to the guidelines provided in the `data/` directory.
2. Adjust the configuration settings in `config.json` as needed.
3. Train the model using:

```bash
python train.py --config config.json
```

4. Evaluate the model with:

```bash
python evaluate.py --checkpoint path/to/your/model.ckpt
```

Refer to the `examples/` directory for more detailed usage examples.

## Contributing

Contributions to CaloPointFlow II are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- Fast Calorimeter Simulation Challenge (CaloChallenge) for providing the datasets.
