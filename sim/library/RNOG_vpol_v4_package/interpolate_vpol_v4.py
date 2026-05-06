"""
Generate an RNO-G VPol v4 pkl at any refractive index via PCHIP interpolation.

Usage:
    python interpolate_vpol_v4.py --n 1.74
    python interpolate_vpol_v4.py --n 1.65 --output my_model.pkl

Output is a standard NuRadioMC antenna pkl, usable with AntennaPattern as-is.

Requires the RNOG_vpol_v4_5inch_center_interp/ directory containing pkl files
at n = 1.0, 1.1, ..., 1.8.
"""
import numpy as np
import pickle
import os
import argparse
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = SCRIPT_DIR


def load_and_interpolate(n, model_dir=DEFAULT_MODEL_DIR):
    """Load all n-value pkls and PCHIP interpolate at target n."""
    pkl_files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pkl') and '_n' in f)
    n_vals, all_Ht, all_Hp = [], [], []
    meta = None

    for pf in pkl_files:
        n_val = float(pf.split('_n')[-1].replace('.pkl', ''))
        with open(os.path.join(model_dir, pf), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if meta is None:
            meta = tuple(data[:4]) + (data[4], data[5], data[6])
        n_vals.append(n_val)
        all_Ht.append(np.array(data[8], dtype=complex))
        all_Hp.append(np.array(data[7], dtype=complex))

    n_vals = np.array(n_vals)
    si = np.argsort(n_vals)
    n_vals = n_vals[si]
    Ht = np.array([all_Ht[i] for i in si])
    Hp = np.array([all_Hp[i] for i in si])

    Ht_out = (PchipInterpolator(n_vals, np.abs(Ht), axis=0)(n)
              * np.exp(1j * PchipInterpolator(n_vals, np.unwrap(np.angle(Ht), axis=0), axis=0)(n)))
    Hp_out = (PchipInterpolator(n_vals, np.abs(Hp), axis=0)(n)
              * np.exp(1j * PchipInterpolator(n_vals, np.unwrap(np.angle(Hp), axis=0), axis=0)(n)))

    return meta[:4] + (meta[4], meta[5], meta[6]), Hp_out, Ht_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=float, required=True)
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--output', type=str, default=None)
    args, _ = parser.parse_known_args()

    meta, Hp, Ht = load_and_interpolate(args.n, args.model_dir)
    out = args.output or f'RNOG_vpol_v4_5inch_center_n{args.n:.2f}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(list(meta) + [Hp, Ht], f, protocol=2)
    print(f'Saved {out} (n={args.n})')
