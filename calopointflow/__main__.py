from .utils import load_config
from .train import train as _train
from .generate import generate as _generate
from .plot import plot as _plot

import click
from omegaconf import OmegaConf, DictConfig
import multiprocessing

# General options decorator
def common_options(func):
    options = [
        click.option('-m', '--model', type=str, required=True, help='Model to use. Options: I, dsf, dsf_cdeq, II'),
        click.option('-d', '--dataset', type=int, required=True, help='Dataset to use. Options: 2, 3')
    ]
    for option in reversed(options):
        func = option(func)
    return func

@click.group()
def cli():
    pass

# debug arguments: train -m II -d 2 -ld ~/beegfs/tmp/
@cli.command()
@common_options
@click.option("-ld", "--log_dir", type=str, required=True, help="Path to save the logs")
@click.argument('kwargs', nargs=-1)  # Accept wildcard arguments
def train(model, dataset, log_dir, **kwargs):
    cfg = _load_config(model, dataset)
    _train(model, dataset, cfg, log_dir)

# debug arguments: generate -m II -d 2 --ckpt_path=/home/schnakes/beegfs/calopointflow_uplift/d2_II_001.ckpt --save_path=~/test.hdf5
@cli.command()
@common_options
@click.option("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file")
@click.option("--save_path", type=str, required=True, help="Path to save the generated data")
@click.argument('kwargs', nargs=-1)  # Accept wildcard arguments
def generate(model, dataset, ckpt_path, save_path, **kwargs):
    cfg = _load_config(model, dataset)
    _generate(cfg, ckpt_path, save_path)

# debug arguments: plot -p marginals -d 2  -g4 /dev/shm/g4_dataset_2.hdf5 -cpf /dev/shm/cpf_dataset_2.hdf5
@cli.command()
@click.option('-d', '--dataset', type=int, required=True, help='Dataset to use. Options: 2, 3')
@click.option("-p", "--plot", type=str, default='all', help="""
              What to plot.\n
              Options:\n
                all: Plot all available plots\n
                marginals: Plot the marginal distributions of the data in the z, alpha, and r dimensions\n
                layer_energies: Plot the energy distributions of the data in individual layer areas\n
                corrcoeff: Plot the correlation coefficients between the data in the z, alpha, and r dimensions\n
                cov_eigenvalues: Plot the histograms of eigenvalues of the covariance matrices of the individual showers\n
                means: Plot the shower means in the z, alpha, and r dimensions\n
                cell_energies: Plot the energy distributions of the data in individual cells\n
                num_hits: Plot the histogram of number of hits\n
              """)
@click.option("--save_path", type=str, default="", help="Path to save the plots")
@click.option("-g4", "--geant4_data", type=str, required=True, help="Path to the Geant4 data")
@click.option("-cpf", "--calopointflow_data", type=str, required=True, help="Path to the CaloPointFlow data")
def plot(dataset, plot, save_path, geant4_data, calopointflow_data):
    _plot(dataset, plot, save_path, geant4_data, calopointflow_data)


def _load_config(model: str, dataset: int) -> DictConfig:
    base_cfg = load_config("base")
    dataset_cfg = load_config(f"d{dataset}_{model}")
    cfg = OmegaConf.merge(base_cfg, dataset_cfg)

    for key, value in OmegaConf.from_cli().items():
        if key.startswith("-"):
            continue
        if value is not None:
            OmegaConf.update(cfg, key, value)

    if cfg.data.num_workers == -1:
        cfg.data.num_workers = multiprocessing.cpu_count() - 1

    return cfg

if __name__ == '__main__':
    cli()