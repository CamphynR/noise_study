"""Generate VPol v4 summary plot."""
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator

all_ns = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
loo_ns = [1.1, 1.3, 1.5, 1.7]

H_all = {}
f_ghz = None
for n in all_ns:
    mn = f"RNOG_vpol_v4_5inch_center_n{n:.2f}"
    with open(f"{mn}.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")
    idx = np.where((np.abs(data[5] - np.pi / 2) < 0.01) & (np.abs(data[6]) < 0.01))[0]
    H_all[n] = data[8][idx]
    if f_ghz is None:
        f_ghz = data[4][idx]

fmask = (f_ghz * 1e3 >= 50) & (f_ghz * 1e3 <= 800)
f_mhz = f_ghz[fmask] * 1e3
H_stack = np.array([H_all[n][fmask] for n in all_ns])

fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
              height_ratios=[1, 1, 1.2])

# Row 0: magnitude, phase, group delay
ax = fig.add_subplot(gs[0, 0])
for n in all_ns:
    ax.plot(f_mhz, np.abs(H_all[n][fmask]) * 100, lw=0.9, label=f"n={n:.1f}")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("|H_theta| (cm)")
ax.set_title("Boresight RVEL magnitude")
ax.set_xlim(50, 800)
ax.set_ylim(0, 25)
ax.legend(fontsize=5, ncol=2)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
for n in [1.0, 1.3, 1.5, 1.7]:
    ax.plot(f_mhz, np.rad2deg(np.unwrap(np.angle(H_all[n][fmask]))), lw=1, label=f"n={n:.1f}")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Phase (deg)")
ax.set_title("Boresight phase")
ax.set_xlim(50, 800)
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
for n in [1.0, 1.3, 1.5, 1.7]:
    phase = np.unwrap(np.angle(H_all[n][fmask]))
    gd = -np.diff(phase) / (2 * np.pi * np.diff(f_ghz[fmask]))
    ax.plot(f_mhz[:-1], gd, lw=0.8, label=f"n={n:.1f}")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Group delay (ns)")
ax.set_title("Boresight group delay")
ax.set_xlim(50, 800)
ax.legend()
ax.grid(True, alpha=0.3)

# Row 1: impulse, multi-zenith, multi-zenith impulse
ax = fig.add_subplot(gs[1, 0])
for n in [1.0, 1.3, 1.5, 1.7]:
    h_td = np.fft.irfft(H_all[n])
    t = np.arange(len(h_td)) * 0.1
    ax.plot(t, h_td * 100, lw=0.8, label=f"n={n:.1f}")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("h (cm)")
ax.set_title("Boresight impulse response")
ax.set_xlim(-0.5, 15)
ax.legend()
ax.grid(True, alpha=0.3)

with open(f"RNOG_vpol_v4_5inch_center_n1.50.pkl", "rb") as f:
    data50 = pickle.load(f, encoding="latin1")

ax = fig.add_subplot(gs[1, 1])
for zd in [90, 70, 50, 30, 10]:
    idx2 = np.where((np.abs(data50[5] - np.deg2rad(zd)) < 0.05) & (np.abs(data50[6]) < 0.01))[0]
    if len(idx2) > 0:
        fm2 = data50[4][idx2]
        mm2 = (fm2 * 1e3 >= 50) & (fm2 * 1e3 <= 800)
        ax.plot(fm2[mm2] * 1e3, np.abs(data50[8][idx2][mm2]) * 100, lw=1, label=f"zen={zd}")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("|H_theta| (cm)")
ax.set_title("n=1.50 at multiple zeniths")
ax.set_xlim(50, 800)
ax.set_ylim(0, 25)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
for zd in [90, 70, 50, 30]:
    idx2 = np.where((np.abs(data50[5] - np.deg2rad(zd)) < 0.05) & (np.abs(data50[6]) < 0.01))[0]
    if len(idx2) > 0:
        h_td2 = np.fft.irfft(data50[8][idx2])
        dt2 = 1.0 / (2 * data50[4][idx2][-1])
        t2 = np.arange(len(h_td2)) * dt2
        ax.plot(t2, h_td2 * 100, lw=0.8, label=f"zen={zd}")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("h (cm)")
ax.set_title("n=1.50 impulse at multiple zeniths")
ax.set_xlim(-0.5, 15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2: leave-one-out stacked (4 rows x 3 cols: mag error, phase error, overlay)
inner = gs[2, :].subgridspec(4, 3, hspace=0.15, wspace=0.3)

for ri, n_held in enumerate(loo_ns):
    i = np.where(all_ns == n_held)[0][0]
    mask = np.arange(len(all_ns)) != i
    mag_i = PchipInterpolator(all_ns[mask], np.abs(H_stack[mask]), axis=0)(n_held)
    ph_i = PchipInterpolator(all_ns[mask], np.unwrap(np.angle(H_stack[mask]), axis=0), axis=0)(n_held)
    H_interp = mag_i * np.exp(1j * ph_i)
    H_true = H_stack[i]
    rmspe_mag = np.abs(mag_i - np.abs(H_true)) / np.maximum(np.abs(H_true), 1e-10) * 100
    rmspe_ph = np.abs(np.angle(H_interp / H_true)) * 180 / np.pi

    ax = fig.add_subplot(inner[ri, 0])
    ax.plot(f_mhz, rmspe_mag, lw=0.8, color=f"C{ri}")
    ax.axhline(1, color="r", ls="--", lw=0.5, alpha=0.5, label="1% target" if ri == 0 else None)
    ax.set_ylim(0, 5)
    ax.set_xlim(50, 800)
    ax.set_ylabel(f"n={n_held:.1f}", fontsize=9, color="red", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if ri == 0:
        ax.set_title("Magnitude error (%)", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
    if ri < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Freq (MHz)", fontsize=8)

    ax = fig.add_subplot(inner[ri, 1])
    ax.plot(f_mhz, rmspe_ph, lw=0.8, color=f"C{ri}")
    ax.axhline(5, color="r", ls="--", lw=0.5, alpha=0.5, label="5 deg target" if ri == 0 else None)
    ax.set_ylim(0, 10)
    ax.set_xlim(50, 800)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if ri == 0:
        ax.set_title("Phase error (deg)", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
    if ri < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Freq (MHz)", fontsize=8)

    ax = fig.add_subplot(inner[ri, 2])
    ax.plot(f_mhz, mag_i * 100, lw=1.5, color=f"C{ri}")
    ax.plot(f_mhz, np.abs(H_true) * 100, color="k", lw=2, ls=(0, (8, 4)), zorder=10)
    ax.set_ylim(0, 25)
    ax.set_xlim(50, 800)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if ri == 0:
        ax.set_title("Interp (color) vs actual (dashed)", fontsize=9)
    if ri < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Freq (MHz)", fontsize=8)

import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText, HPacker, TextArea

# Border around bottom row
rect = mpatches.FancyBboxPatch((0.1, 0.08), 0.81, 0.28,
                                boxstyle="round,pad=0.01",
                                facecolor="none", edgecolor="gray",
                                linewidth=1.5, transform=fig.transFigure,
                                clip_on=False)
fig.patches.append(rect)

# Label box sitting on top edge of border
bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", linewidth=1.5)
t1 = TextArea("Leave-one-out interpolation validation  |  ",
              textprops=dict(fontsize=11, fontstyle="italic"))
t2 = TextArea("held-out n",
              textprops=dict(fontsize=11, fontstyle="italic", color="red", fontweight="bold"))
t3 = TextArea("  labeled at left of each row",
              textprops=dict(fontsize=11, fontstyle="italic"))
packed = HPacker(children=[t1, t2, t3], pad=0, sep=0, align="baseline")

from matplotlib.offsetbox import AnchoredOffsetbox
ab = AnchoredOffsetbox(loc="upper center", child=packed, pad=0,
                       bbox_to_anchor=(0.5, 0.38), bbox_transform=fig.transFigure,
                       frameon=True, prop=dict(size=11))
ab.patch.set_boxstyle("round,pad=0.4")
ab.patch.set_facecolor("white")
ab.patch.set_edgecolor("gray")
ab.patch.set_linewidth(1.5)
fig.add_artist(ab)

fig.suptitle("VPol v4 summary", fontsize=30, y=0.95)
outpath = "vpol_v4_summary.png"
fig.savefig(outpath, dpi=150)
print(f"Saved {outpath}")
