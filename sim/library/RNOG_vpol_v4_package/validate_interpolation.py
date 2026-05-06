"""
Leave-one-out cross-validation of VPol v4 PCHIP interpolation.

Holds out each interior n value (1.2-1.7), interpolates from the rest,
and compares magnitude and phase at boresight. Produces a validation plot.

Usage:
    python validate_interpolation.py
"""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_boresight(model_dir):
    """Load all pkl files and extract boresight H_theta."""
    pkl_files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pkl') and '_n' in f)
    n_values, H_list, freqs_out = [], [], None

    for pf in pkl_files:
        n_val = float(pf.split('_n')[-1].replace('.pkl', ''))
        with open(os.path.join(model_dir, pf), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        freqs, thetas, phis, H_theta = data[4], data[5], data[6], data[8]
        idx = np.where((np.abs(thetas - np.pi/2) < 0.01) & (np.abs(phis) < 0.01))[0]
        n_values.append(n_val)
        H_list.append(H_theta[idx])
        if freqs_out is None:
            freqs_out = freqs[idx]

    n_values = np.array(n_values)
    si = np.argsort(n_values)
    return n_values[si], np.array([H_list[i] for i in si]), freqs_out


def align_onsets(ns, H_stack, freqs):
    """Apply onset alignment across n values."""
    for k in range(len(ns)):
        h_td = np.fft.irfft(H_stack[k])
        pk = np.max(np.abs(h_td))
        if pk > 0:
            onset = np.argmax(np.abs(h_td) > 0.01 * pk)
            dt = 1.0 / (2 * freqs[-1])
            H_stack[k] *= np.exp(1j * 2 * np.pi * freqs * onset * dt)
    return H_stack


def main():
    ns, H_stack, freqs = load_all_boresight(SCRIPT_DIR)
    H_stack = align_onsets(ns, H_stack, freqs)
    f_mhz = freqs * 1e3

    held_out = [n for n in ns if 1.15 < n < 1.75]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: all simulated n values
    ax = axes[0]
    for i, n in enumerate(ns):
        ax.plot(f_mhz, np.abs(H_stack[i]) * 100, linewidth=1.2, label=f'n={n:.1f}')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('|H_theta| (cm)')
    ax.set_title('Boresight H_theta: all simulated n values')
    ax.set_xlim(50, 800)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: leave-one-out
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(held_out)))
    print(f'Leave-one-out validation (boresight, 50-800 MHz):')
    print(f'{"n":>5s}  {"mag_mean%":>9s}  {"mag_95%":>8s}  {"ph_mean":>8s}  {"ph_95":>8s}')

    for ci, n_held in enumerate(held_out):
        mask = ns != n_held
        mag_i = PchipInterpolator(ns[mask], np.abs(H_stack[mask]), axis=0)(n_held)
        ph_i = PchipInterpolator(ns[mask], np.unwrap(np.angle(H_stack[mask]), axis=0), axis=0)(n_held)
        H_interp = mag_i * np.exp(1j * ph_i)

        i_held = np.where(ns == n_held)[0][0]
        H_true = H_stack[i_held]

        valid = np.abs(H_true) > 1e-6
        mag_err = np.abs(np.abs(H_interp[valid]) - np.abs(H_true[valid])) / np.abs(H_true[valid]) * 100
        ph_err = np.abs(np.angle(H_interp[valid] / H_true[valid])) * 180 / np.pi
        print(f'{n_held:>5.1f}  {np.mean(mag_err):>9.3f}  {np.percentile(mag_err, 95):>8.3f}  {np.mean(ph_err):>7.2f}d  {np.percentile(ph_err, 95):>7.2f}d')

        ax.plot(f_mhz, mag_i * 100, '-', color=colors[ci], linewidth=2.5)
        ax.plot(f_mhz, np.abs(H_true) * 100, '--', color='k', linewidth=1, zorder=10)

    ax.plot([], [], '-', color='C0', linewidth=2.5, label='Interpolated (colored)')
    ax.plot([], [], 'k--', linewidth=1, label='Actual held-out (black dashed)')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('|H_theta| (cm)')
    ax.set_title(f'Leave-one-out: {", ".join(f"n={n:.1f}" for n in held_out)}')
    ax.set_xlim(50, 800)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('VPol v4 boresight RVEL: simulations and leave-one-out validation', fontsize=13)
    fig.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'validation_leave_one_out.png')
    fig.savefig(outpath, dpi=150)
    print(f'\nSaved {outpath}')


if __name__ == '__main__':
    main()
