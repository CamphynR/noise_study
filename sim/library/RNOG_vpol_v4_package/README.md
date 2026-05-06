# RNO-G VPol v4 antenna models with refractive index interpolation

XFdtd simulations of the RNO-G VPol fat dipole in an 11.2" borehole at 9 refractive index values (n = 1.0 to 1.8 in steps of 0.1). Each pkl file is a standard NuRadioMC antenna model.

## Quick start

Generate a pkl at your desired refractive index:

```bash
python interpolate_vpol_v4.py --n 1.74
```

This creates `RNOG_vpol_v4_5inch_center_n1.74.pkl` in the current directory.

## Using it with NuRadioMC

Copy the output pkl into your NuRadioMC AntennaModels directory:

```bash
MODEL=RNOG_vpol_v4_5inch_center_n1.74
mkdir -p /path/to/NuRadioMC/NuRadioReco/detector/AntennaModels/$MODEL
cp ${MODEL}.pkl /path/to/NuRadioMC/NuRadioReco/detector/AntennaModels/$MODEL/
```

Then reference `RNOG_vpol_v4_5inch_center_n1.74` as the antenna model in your detector description.

## Constraints

- **Interpolation range:** n = 1.0 to 1.8. Do not extrapolate outside this range.
- **Recommended range:** n = 1.1 to 1.7. Interior points have the best accuracy. Edge values (n = 1.0, 1.8) have larger errors.
- **Angular coverage:** 37 zenith angles (0-180 deg, 5 deg steps) x 73 azimuth angles (0-360 deg, 5 deg steps). The base simulations have a single azimuth (phi=0) and zenith 0-90 deg. The full sphere is built by assuming perfect azimuthal symmetry (same response at all phi) and mirroring about zenith=90 deg. The VPol fat dipole is azimuthally symmetric in reality so this should be a fair assumption.
- **Frequency range:** 0 to 1000 MHz in 2 MHz steps (501 bins). The RVEL magnitude drops below ~1 cm above 600-700 MHz, so phase and group delay become less reliable at those frequencies.

## Processing steps

The base simulations are from XFdtd (Dan Smith, VPol v2 series) at 9 refractive index values. The following processing was applied to produce these pkls:

1. **FFT and normalization:** Time-domain impulse responses were FFTed and divided by 2 to convert from the two-sided to one-sided spectrum convention used in NuRadioMC pkl files.

2. **Onset time alignment:** The simulation time axis starts at -50 ns, but the FFT assumes sample 0 is at t=0. Without correcting for this, the impulse response is aphysical (starts at a large nonzero value instead of rising from zero) and the phase contains a large linear ramp (~300 rad/GHz) corresponding to ~48 ns of artificial group delay. The correction applies a frequency-domain phase shift of `exp(j * 2*pi*f * onset_sample * dt)` where `onset_sample` is the index of the first significant sample in the time-domain data. This also removes an alternating 0.1 ns offset between even/odd n values (from integer-sample rounding in the original processing), which otherwise caused ~16 degree mean phase interpolation errors across n. After correction, interpolation phase errors are below 1.1 degrees for interior n values.

3. **Frequency upsampling:** The native frequency resolution is 10 MHz. At this resolution, NuRadioMC's trilinear interpolation between frequency bins introduces visible ripples in the group delay (the derivative of phase amplifies small interpolation errors). The data was upsampled to 2 MHz via cubic interpolation of magnitude and unwrapped phase separately, which eliminates these ripples and matches the resolution of the v3 antenna models.

4. **Full-sphere expansion:** The simulations cover zenith 0-90 deg and azimuth 0 deg only. The data was expanded to the full sphere (0-180 deg zenith, 0-360 deg azimuth) using dipole symmetry: the response is mirrored about zenith = 90 deg and is azimuthally symmetric.

## Interpolation method

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation of magnitude and unwrapped phase separately across the 9 n values at each (frequency, zenith, azimuth) bin.

## Interpolation accuracy

Leave-one-out cross-validation at boresight (50-800 MHz):

| Held-out n | Mag error (mean) | Mag error (95th pct) | Phase error (mean) | Phase error (95th pct) |
|-----------|-----------------|---------------------|-------------------|----------------------|
| 1.1 (edge) | 0.96% | 4.1% | 3.1 deg | 6.5 deg |
| 1.3 | 0.20% | 0.79% | 0.97 deg | 1.8 deg |
| 1.5 | 0.09% | 0.34% | 0.95 deg | 1.7 deg |
| 1.7 | 0.13% | 0.78% | 1.0 deg | 1.8 deg |
| 1.8 (edge) | 0.35% | 1.9% | 2.9 deg | 5.2 deg |

## Summary plot

See `vpol_v4_summary.png` for a 9-panel overview:

- **Row 1:** Boresight RVEL magnitude (all n values), boresight phase (selected n), boresight group delay (selected n)
- **Row 2:** Boresight impulse response (selected n), n=1.50 RVEL at multiple zenith angles, n=1.50 impulse at multiple zeniths
- **Row 3:** Leave-one-out validation for n = 1.1, 1.3, 1.5, 1.7 showing magnitude error (%), phase error (deg), and interpolated vs actual RVEL overlay

Run `python validate_interpolation.py` to reproduce the validation.

## Options

```
python interpolate_vpol_v4.py --n 1.65                    # default output name
python interpolate_vpol_v4.py --n 1.74 --output my.pkl    # custom output path
python interpolate_vpol_v4.py --n 1.74 --model_dir /path  # custom pkl directory
```

## Contents

| File | Description |
|------|-------------|
| `RNOG_vpol_v4_5inch_center_n*.pkl` | 9 base antenna models (n = 1.0, 1.1, ..., 1.8) |
| `interpolate_vpol_v4.py` | Interpolation script |
| `validate_interpolation.py` | Leave-one-out cross-validation script |
| `vpol_v4_summary.png` | Summary plot |
| `README.md` | This file |

## Requirements

- numpy
- scipy (PchipInterpolator)
- matplotlib (for validation script only)
