import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

import  mplhep as hep

COLOR_GEN = '#0082c8'
COLOR_VAL = '#e6194b'

# Set the style
hep.style.use("CMS")
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams['font.family'] = "STIXGeneral"

fig, ax = plt.subplots(figsize=(8, 6))

low_level = [0.966, 0.960, 0.892, 0.874]
low_level_err = [0.001, 0.001, 0.002, 0.004]
high_level = [0.944, 0.928, 0.881, 0.8328]
high_level_err = [0.001, 0.002, 0.0034, 0.0024]


ax.errorbar([1, 2, 3, 4], low_level, yerr=np.array(low_level_err) * np.sqrt(10), fmt='-o', markersize=3.5, linewidth=1, capsize=2,  color=COLOR_GEN, label='low-level')
ax.errorbar([1, 2, 3, 4], high_level, yerr=np.array(low_level_err) * np.sqrt(10), fmt='-o', markersize=3.5, linewidth=1, capsize=2, color=COLOR_VAL, label='high-level')

#ax.errorbar([1, 2, 3, 4], high_level, yerr=np.array(low_level_err) * np.sqrt(10), color=COLOR_VAL, label='high-level')

ax.set_xticks([1, 2, 3, 4], labels=["I", "+DSF", "+3d", "II"], minor=False)
ax.legend()
ax.set_ylabel("AUC")
ax.set_xlabel("Ablation study")
fig.tight_layout()

fig.savefig("ablation.pdf")
