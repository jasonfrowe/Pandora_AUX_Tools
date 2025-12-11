# Pandora Auxiliary Targets Data Processing

This repository contains tools for the Pandora mission, including notebooks for processing auxiliary target lists and simulating crowded field observations.

## Overview

### Aux_Targets.ipynb
The auxiliary targets processing notebook performs the following operations:
1. Loads the base target CSV file
2. Imports AP (chemically peculiar) stars from the CDS catalog
3. Fills missing stellar data (coordinates, proper motions, J magnitudes) from SIMBAD
4. Fills missing planet orbital parameters from NASA Exoplanet Archive
5. Enriches proper motions using Gaia DR3
6. Saves the final enriched dataset

### Scene_sim.ipynb
The scene simulator generates synthetic Pandora images of crowded stellar fields:
- Queries Gaia DR2 for stellar field around target
- Generates stellar spectra using exotic_ld limb darkening library
- Simulates spectral traces for all stars in field of view
- Models position angle variations (0-360° in 5° steps)
- Computes spectral dilution from nearby contaminating sources
- Produces field images and dilution maps

**Status**: Functional with shared pixel array optimization providing ~10% performance improvement. Runtime approximately 52 seconds per position angle (72 total angles). Main performance bottleneck is expensive exotic_ld stellar spectrum calculations (called 530 times per iteration)

## Features

- **Idempotent Design**: Can be run multiple times safely without duplicating data
- **Smart Querying**: Skips rows that already have populated fields
- **Duplicate Detection**: Won't re-import AP stars if already present
- **Batch Processing**: Gaia queries are batched for efficiency
- **Multiple Format Support**: Handles both letter-based (b, c, d) and numeric (0.01, 0.02) planet designations

## Requirements

### Aux_Targets.ipynb
```bash
pip install pandas numpy astropy astroquery
```

### Scene_sim.ipynb
```bash
pip install numpy scipy matplotlib astropy astroquery tqdm numba exotic_ld
```
Note: `exotic_ld` requires stellar atmosphere data files in `exotic_ld_data/` directory

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy astropy astroquery ipykernel
   ```
4. Install the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=pandora-venv --display-name="Python (Pandora .venv)"
   ```

## Usage

1. Place your input CSV file in the repository directory
2. Update the filename in the notebook (cell 5) if needed
3. Run all cells in order
4. The enriched data will be saved as `Pandora Snapshot Targets - Targets_updated.csv`

## Data Sources

- **SIMBAD**: Stellar coordinates, proper motions, and J magnitudes
- **NASA Exoplanet Archive**: Planet orbital periods and transit ephemerides
- **Gaia DR3**: Proper motions and refined astrometry for stars with Gaia IDs
- **CDS (ApJ/943/147)**: AP star catalog

## File Structure

```
Pandora/
├── Aux_Targets.ipynb           # Target list processing notebook
├── Scene_sim.ipynb             # Crowded field scene simulator
├── pandora.py                  # Core simulation functions
├── Pandora Snapshot Targets - *.csv  # Input CSV files
├── Figures/                    # Output figures directory
├── exotic_ld_data/             # Stellar atmosphere library
├── Kernels/                    # PSF kernels
├── PSG/                        # Planet spectra
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Notes

- AP stars are skipped during SIMBAD queries (they get data from Gaia instead)
- The notebook tracks progress and shows counts of updated/skipped rows
- Final output includes data completeness statistics

## License

This project is licensed under the GNU General Public License (GPL) version 3 or later.

## Contact

Jason Rowe - jason@jasonrowe.org
