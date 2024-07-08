import h5py
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import  mplhep as hep
import os
import torch

COLOR_GEN = '#0082c8'
COLOR_VAL = '#e6194b'

def plot(dataset, plot_cmd, save_path, g4_data_path, cpf_data_path):
    # Set the style
    hep.style.use("CMS")
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams['font.family'] = "STIXGeneral"

    data = {}
    data["g4"] = _load_data(g4_data_path)
    data["cpf"] = _load_data(cpf_data_path)
    data["dataset"] = dataset

    if dataset == 2:
        data["size"] = (45, 16, 9)
    elif dataset == 3:
        data["size"] = (45, 50, 18)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data["g4"]["showers"] = np.reshape(data["g4"]["showers"], (data["g4"]["showers"].shape[0], *data["size"])) 
    data["cpf"]["showers"] = np.reshape(data["cpf"]["showers"], (data["cpf"]["showers"].shape[0], *data["size"]))

    # Register of all plots
    plots = {
        "marginals": plot_marginals,
        "layer_energies": plot_layer_energies, 
        "corrcoeff": plot_correlation_coefficients,
        "cov_eigenvalues": plot_cov_eigenvalues,
        "means": plot_means,
        "cell_energies": plot_cell_energies,
        "num_hits": plot_num_hits,
    }

    if plot_cmd == "all":
        for  plot_cmd in plots:
            plots[plot_cmd](data, save_path)
    elif plot_cmd in plots:
        plots[plot_cmd](data, save_path)
    else:
        raise ValueError(f"Unknown plot command: {plot_cmd}")
        
def plot_marginals(data, save_path):

    z_vals = {
        "axis" : (1, 2),
        "bins": data["size"][0],
        "ticks_width": 5,
        "x_label": "z",
        "y_lim": (0, 8.5),
    }

    if data["dataset"] == 2:

        alpha_vals = {
            "axis" : (0, 2),
            "bins": data["size"][1],
            "ticks_width": 5,
            "x_label": r"$\alpha$",
            "y_lim": (0, 11),
        }

        r_vals = {
            "axis" : (0, 1),
            "bins": data["size"][2],
            "ticks_width": 2,
            "x_label": "r",
            "y_lim": (0, 75),
        }

    elif data["dataset"] == 3:

        alpha_vals = {
            "axis" : (0, 2),
            "bins": data["size"][1],
            "ticks_width": 5,
            "x_label": r"$\alpha$",
            "y_lim": (0, 3.5)
        }

        r_vals = {
            "axis" : (0, 1),
            "bins": data["size"][2],
            "ticks_width": 2,
            "x_label": "r",
            "y_lim": (0, 50)
        }
    else:
        raise ValueError(f"Unknown dataset: {data['dataset']}")
    

    data["cpf"]["shower_mean"] = np.mean(data["cpf"]["showers"], axis=0) 
    data["g4"]["shower_mean"] = np.mean(data["g4"]["showers"], axis=0)

    fig, axis = plt.subplots(2, 3, figsize=(18, 6), gridspec_kw={'height_ratios': [3, 1]})

    for (i, val) in enumerate([z_vals, alpha_vals, r_vals]):

        x_cpf = np.sum(data["cpf"]["shower_mean"], axis=val["axis"])
        x_g4 = np.sum(data["g4"]["shower_mean"], axis=val["axis"])

        bins = np.arange(0.5, val["bins"] + 1.5)

        ax = axis[0, i]
        ax.stairs(x_g4 / 1000, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
        ax.stairs(x_cpf / 1000, bins, label="CPF", color=COLOR_GEN, linewidth=2)
        ax.set_ylabel("Energy [GeV]")
        ax.legend()
        ax.set_xlim(0, val["bins"]+1)
        ax.set_xticks(np.arange(1, val["bins"]+1, val["ticks_width"]), labels=[])
        ax.set_ylim(val["y_lim"])
        if data["dataset"] == 2:
            ax.set_title("Dataset II", loc="right", fontsize=18)
        elif data["dataset"] == 3:
            ax.set_title("Dataset III", loc="right", fontsize=18)

        bins_center  = (bins[1:] + bins[:-1]) / 2
        diff_cp = (x_cpf - x_g4)/x_g4
        err_cp = _error_diff(x_cpf, x_g4)

        non_zero = (x_cpf != 0) & (x_g4 != 0)

        ax = axis[1, i]
        ax.errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)

        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel(val["x_label"])
        ax.set_ylabel(r"$\Delta$CPF/G4")
        ax.set_xlim(0, val["bins"]+1)

        ax.plot([-1000, 1000], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pth = os.path.join(save_path, f"shower_shape_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)

def plot_layer_energies(data, save_path):
    x_cpf = np.sum(data["cpf"]["showers"], axis=(2, 3))
    x_g4 = np.sum(data["g4"]["showers"], axis=(2, 3))

    fig, axis = plt.subplots(2, 5, figsize=(30, 6), gridspec_kw={'height_ratios': [3, 1]})

    dict1 = {
        "energy_max": 55,
        "2": 1,
    }

    dict2 = {
        "energy_max": 65,
        "2": 0.5,
    }

    dict3 = {
        "energy_max": 55,
        "diff_max": 0.5,
    }

    dict4 = {
        "energy_max": 25,
        "diff_max": 2,
    }

    dict5 = {
        "energy_max": 4,
        "2": 10,
    }

    for i, d in enumerate([dict1, dict2, dict3, dict4, dict5]):
        energy_max = d["energy_max"]
        bins = 10 ** np.linspace(-2, np.log10(energy_max), 50)
        start = i * 9
        end = (i + 1) * 9

        x_g4_hist, _  = np.histogram(x_g4[:, start:end].flatten() / 1000, bins=bins)
        x_cpf_hist, _ = np.histogram(x_cpf[:, start:end].flatten() / 1000, bins=bins)

        ax = axis[0, i]
        ax.stairs(x_g4_hist, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
        ax.stairs(x_cpf_hist, bins, label="CPF", color=COLOR_GEN, linewidth=2)

        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel("Entries")
        ax.set_ylim(1, 5e6)
        ax.set_xlim(1e-2 / 2, energy_max*2)
        ax.set_title(f"Layer {start+1}-{end} in z", loc="left")
        if data["dataset"] == 2:
            ax.set_title("Dataset II", loc="right")
        elif data["dataset"] == 3:
            ax.set_title("Dataset III", loc="right")

        bins_center  = (bins[1:] + bins[:-1]) / 2
        diff_cp = (x_cpf_hist - x_g4_hist)/x_g4_hist
        err_cp = _error_diff(x_cpf_hist, x_g4_hist)

        non_zero = (x_cpf_hist != 0) & (x_g4_hist != 0)

        ax = axis[1, i]
        ax.errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)

        ax.set_ylim(-1.25, 1.25)
        ax.set_xscale('log')
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"$\Delta$CPF/G4")
        ax.set_xlim(1e-2 / 2, energy_max*2)
        ax.plot([1e-2 / 2, energy_max*2], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pth = os.path.join(save_path, f"energy_layers_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)


def plot_correlation_coefficients(data, save_path):
    s_cpf = torch.from_numpy(data["cpf"]["showers"]).float()
    s_g4 = torch.from_numpy(data["g4"]["showers"]).float()

    s_cpf = s_cpf.view(s_cpf.size(0), -1).transpose(0, 1)
    s_g4 = s_g4.view(s_g4.size(0), -1).transpose(0, 1)

    corrcoef_cpf = torch.corrcoef(s_cpf).numpy()
    corrcoef_g4 = torch.corrcoef(s_g4).numpy() 

    fig, axis = plt.subplots(1, 3, figsize=(18, 6))

    _plot_corrcoef(corrcoef_cpf, fig, axis[0], "CPF", *data["size"])
    _plot_corrcoef(corrcoef_g4, fig, axis[1], "Geant4", *data["size"])
    _plot_corrcoef(corrcoef_cpf - corrcoef_g4, fig, axis[2], r"$\Delta$CPF", *data["size"])

    fig.set_tight_layout(True)
    pth = os.path.join(save_path, f"corrcoeff_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)

def plot_cov_eigenvalues(data, save_path):
    _calc_means_and_eigenvalues(data)

    fig, ax = plt.subplots(2, 3, figsize=(18, 6), gridspec_kw={'height_ratios': [3, 1]})

    if data["dataset"] == 2:
        dict1 = {
            "bin_max": 80,
            "diff_max": 5,
        }

        dict2 = {
            "bin_max": 35,
            "diff_max": 5,
        }

        dict3 = {
            "bin_max": 30,
            "diff_max": 5,
        }

    elif data["dataset"] == 3:

        dict1 = {
            "bin_max": 320,
            "diff_max": 5,
        }

        dict2 = {
            "bin_max": 100,
            "diff_max": 5,
        }

        dict3 = {
            "bin_max": 300,
            "diff_max": 5,
        }
    else:
        raise ValueError(f"Unknown dataset: {data['dataset']}")

    for i, d in enumerate([dict1, dict2, dict3]):
        bin_max = d["bin_max"]

        bins  = np.linspace(0, bin_max, 50)

        g4_hist = np.histogram(data["g4"]["cov_eig"][:, i], bins=bins)[0]
        cpf_hist = np.histogram(data["cpf"]["cov_eig"][:, i], bins=bins)[0]
        

        ax[0, i].stairs(g4_hist, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
        ax[0, i].stairs(cpf_hist, bins, label="CPF", color=COLOR_GEN, linewidth=2)
        ax[0, i].legend()
        #ax[0, i].set_yscale('log')
        ax[0, i].set_ylabel("Entries")
        ax[0, i].set_xlim(bins[0], bins[-1])

        if data["dataset"] == 2:
            ax[0, i].set_title("Dataset II", loc="right", fontsize=18)
        elif data["dataset"] == 3:
            ax[0, i].set_title("Dataset III", loc="right", fontsize=18)

        bins_center  = (bins[1:] + bins[:-1]) / 2
        diff_cp = (cpf_hist - g4_hist)/g4_hist
        err_cp = _error_diff(cpf_hist, g4_hist)

        non_zero = (cpf_hist != 0) & (g4_hist != 0)

        ax[1, i].errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)
        ax[1, i].set_yticks([-2, 0, 2])
        ax[1, i].set_ylim(-2.5, 2.5)
        ax[1, i].set_xlabel(f"Cov Eigenvalue {i+1}")
        ax[1, i].set_ylabel(r"$\Delta$CPF/G4")
        ax[1, i].set_xlim(bins[0], bins[-1])
        ax[1, i].plot([bins[0], bins[-1]], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pth = os.path.join(save_path, f"cov_eigenvalues_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)

def plot_means(data, save_path):
    _calc_means_and_eigenvalues(data)

    if data["dataset"] == 2:
        dict1 = {
            "bin_min": 0,
            "bin_max": 22,
            "dim": "z"
        }

        dict2 = {
            "bin_min": 5,
            "bin_max": 10,
            "dim": r"$\alpha$"
        }

        dict3 = {
            "bin_min": 0.,
            "bin_max": 3.,
            "diff_max": 15,
            "dim": "r"
        }

    elif data["dataset"] == 3:
        dict1 = {
            "bin_min": 0,
            "bin_max": 22,
            "dim": "z"
        }

        dict2 = {
            "bin_min": 15,
            "bin_max": 35,
            "dim": r"$\alpha$"
        }

        dict3 = {
            "bin_min": 0.,
            "bin_max": 5.,
            "dim": "r"
        }
    else:
        raise ValueError(f"Unknown dataset: {data['dataset']}")

    fig, ax = plt.subplots(2, 3, figsize=(18, 6), gridspec_kw={'height_ratios': [3, 1]})

    for i, d in enumerate([dict1, dict2, dict3]):
        bin_min = d["bin_min"]
        bin_max = d["bin_max"]
        dim = d["dim"]

        bins  = np.linspace(bin_min, bin_max, 50)

        cpf_hist = np.histogram(data["cpf"]["means"][:, i], bins=bins)[0]
        g4_hist = np.histogram(data["g4"]["means"][:, i], bins=bins)[0]
        
        ax[0, i].stairs(g4_hist, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
        ax[0, i].stairs(cpf_hist, bins, label="CPF", color=COLOR_GEN, linewidth=2)
        ax[0, i].legend()
        #ax[0, i].set_yscale('log')
        ax[0, i].set_ylabel("Entries")
        ax[0, i].set_xlim(bins[0], bins[-1])
        if data["dataset"] == 2:
            ax[0, i].set_title("Dataset II", loc="right", fontsize=18)
        elif data["dataset"] == 3:
            ax[0, i].set_title("Dataset III", loc="right", fontsize=18)

        bins_center  = (bins[1:] + bins[:-1]) / 2
        diff_cp = (cpf_hist - g4_hist)/g4_hist
        err_cp = _error_diff(cpf_hist, g4_hist)

        non_zero = (cpf_hist != 0) & (g4_hist != 0)

        ax[1, i].errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)
        ax[1, i].set_ylim(-2.5, 2.5)
        ax[1, i].set_yticks([-2, 0, 2])
        ax[1, i].set_xlabel(f"Shower mean in {dim}")
        ax[1, i].set_ylabel(r"$\Delta$CPF/G4")
        ax[1, i].set_xlim(bins[0], bins[-1])
        ax[1, i].plot([bins[0], bins[-1]], [0, 0], 'k', alpha=0.25)
        ax[1, i].plot([bins[0], bins[-1]], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pth = os.path.join(save_path, f"shower_means_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)


def plot_cell_energies(data, save_path):
    val_cpf = data["cpf"]["showers"][data["cpf"]["showers"] > 0]
    val_g4 = data["g4"]["showers"][data["g4"]["showers"] > 0]

    if data["dataset"] == 2:
        bins = 10 ** np.linspace(-2.5, 4.5, 100)
    elif data["dataset"] == 3:   
        bins = 10 ** np.linspace(-2, 2, 50)
    else:
        raise ValueError(f"Unknown dataset: {data['dataset']}")

    cpf_hist, _ = np.histogram(val_cpf, bins=bins)
    g4_hist, _ = np.histogram(val_g4, bins=bins)

    fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})

    ax[0].stairs(g4_hist, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
    ax[0].stairs(cpf_hist, bins, label="CPF", color=COLOR_GEN, linewidth=2)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend()
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].set_ylim(1, 1e8)
    if data["dataset"] == 2:
        ax[0].set_title("Dataset II", loc="right", fontsize=18)
    elif data["dataset"] == 3:
        ax[0].set_title("Dataset III", loc="right", fontsize=18)

    ax[0].set_ylabel("Entries")

    bins_center  = (bins[1:] + bins[:-1]) / 2
    diff_cp = (cpf_hist - g4_hist)/(np.clip(g4_hist, 1e-6, None))
    err_cp = _error_diff(cpf_hist, g4_hist)

    non_zero = (cpf_hist != 0) & (g4_hist != 0)

    ax[1].errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)
    ax[1].set_ylim(-0.6, 0.6)
    ax[1].set_yticks([-0.5, 0, 0.5])
    ax[1].set_xlabel("Cell Energy [MeV]")
    ax[1].set_ylabel(r"$\Delta$CPF/G4")
    ax[1].set_xscale('log')
    ax[1].set_xlim(bins[0], bins[-1])
    ax[1].plot([bins[0], bins[-1]], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pt = os.path.join(save_path, f"cell_energy_dataset_{data['dataset']}.pdf")
    fig.savefig(pt)


def plot_num_hits(data, save_path):
    if data["dataset"] == 2:
        bins = np.arange(-200, 5500, 120)
    elif data["dataset"] == 3:
        bins = np.linspace(-1000, 18500, 120)

    n_hits_g4 = np.sum(data["g4"]["showers"] > 0, axis=(1, 2, 3))
    n_hits_cpf = np.sum(data["cpf"]["showers"] > 0, axis=(1, 2, 3))

    cpf_hist, _ = np.histogram(n_hits_cpf, bins=bins)
    g4_hist, _ = np.histogram(n_hits_g4, bins=bins)

    fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})

    ax[0].stairs(g4_hist, bins, label="Geant4", hatch='//', color=COLOR_VAL, alpha=0.8)
    ax[0].stairs(cpf_hist, bins, label="CPF", color=COLOR_GEN, linewidth=2)
    ax[0].legend()
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].set_title("Dataset III", loc="right", fontsize=18)

    bins_center  = (bins[1:] + bins[:-1]) / 2
    diff_cp = (cpf_hist - g4_hist)/(g4_hist + 1e-6)
    err_cp = _error_diff(cpf_hist, g4_hist)

    non_zero = (cpf_hist != 0) & (g4_hist != 0)

    ax[1].errorbar(bins_center[non_zero], diff_cp[non_zero], err_cp[non_zero], fmt='.', color=COLOR_GEN, markersize=4)

    ax[1].set_ylim(-0.6, 0.6)
    ax[1].set_yticks([-0.5, 0, 0.5])
    ax[1].set_xlabel("Number of Hits")
    ax[1].set_ylabel(r"$\Delta$CPF/G4")
    ax[1].set_xlim(bins[0], bins[-1])
    ax[1].plot([bins[0], bins[-1]], [0, 0], 'k', alpha=0.25)

    fig.tight_layout()
    pth = os.path.join(save_path, f"num_hits_dataset_{data['dataset']}.pdf")
    fig.savefig(pth)





def _calc_means_and_eigenvalues(data):
    idx_cpf = np.argwhere(data["cpf"]["showers"] > 0)
    _, n_cpf = np.unique(idx_cpf[:, 0], return_counts=True)
    idx_cpf = idx_cpf[:, 1:]
    val_cpf = data["cpf"]["showers"][data["cpf"]["showers"] > 0]

    idx_g4 = np.argwhere(data["g4"]["showers"] > 0)
    _, n_g4 = np.unique(idx_g4[:, 0], return_counts=True)
    idx_g4 = idx_g4[:, 1:]
    val_g4 = data["g4"]["showers"][data["g4"]["showers"] > 0]

    cov_g4_eig = []
    means_g4 = []

    cov_cpf_eig = []
    means_cpf = []

    i = 0
    for n in n_cpf:
        idx = idx_cpf[i:i+n]
        val = val_cpf[i:i+n]
        i += n
        x_mean = (val[:, None] * idx).sum(axis=0) / val.sum()
        idx_hat = idx - x_mean
        cov = (val[:, None] * idx_hat).T @ idx_hat / (val.sum() - 1)
        cov_cpf_eig.append(np.linalg.eig(cov)[0])
        means_cpf.append(x_mean)

    i = 0
    for n in n_g4:
        idx = idx_g4[i:i+n]
        val = val_g4[i:i+n]
        i += n
        x_mean = (val[:, None] * idx).sum(axis=0) / val.sum()
        idx_hat = idx - x_mean
        cov = (val[:, None] * idx_hat).T @ idx_hat / (val.sum() - 1)
        cov_g4_eig.append(np.linalg.eig(cov)[0])
        means_g4.append(x_mean)

    data["cpf"]["cov_eig"] = np.array(cov_cpf_eig)
    data["g4"]["cov_eig"] = np.array(cov_g4_eig)

    data["cpf"]["means"] = np.array(means_cpf)
    data["g4"]["means"] = np.array(means_g4)


def _plot_corrcoef(corrcoef, fig, ax, title, num_z, num_alpha, num_r):
    im = ax.matshow(corrcoef, cmap='seismic', vmin=-1, vmax=1)
    ax.set_title(title, loc="left", fontsize=16)
    ax.set_title("Dataset II", loc="right", fontsize=16)
    ticks = np.linspace(0, num_z*num_alpha*num_r, 6).astype(int)
    labels=[f"({(i // (num_alpha * num_r))},{((i // num_r) % num_alpha)},{(i % num_r)})" for i in ticks]
    ax.set_xticks(
        ticks,
        labels=labels,
        #rotation=45,
        ha="center",
        rotation_mode="anchor",
        fontsize=12,
    )
    ax.set_yticks(
        ticks,
        labels=labels,
        #rotation=45,
        ha="right",
        fontsize=12,
    )
    
    fig.colorbar(im, orientation='horizontal', shrink=0.65, pad=0.05, ax=ax)  

def _load_data(path):
    with h5py.File(path, 'r') as f:
        showers = f['showers'][:]
        incident_energies = f['incident_energies'][:]
    
    return {
        "showers": showers,
        "incident_energies": incident_energies
    }

def _error_diff(a, b) :
    return np.sqrt(
        (a / b ** 2) * (1 + a / b)
    )
    
