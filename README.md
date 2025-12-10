# Pandora Auxiliary Targets Data Processing

This repository contains a Jupyter notebook for processing and enriching the Pandora mission's auxiliary target list with data from multiple astronomical databases.

## Overview

The notebook performs the following operations:
1. Loads the base target CSV file
2. Imports AP (chemically peculiar) stars from the CDS catalog
3. Fills missing stellar data (coordinates, proper motions, magnitudes) from SIMBAD
4. Fills missing planet orbital parameters from NASA Exoplanet Archive
5. Enriches proper motions using Gaia DR3
6. Saves the final enriched dataset

## Features

- **Idempotent Design**: Can be run multiple times safely without duplicating data
- **Smart Querying**: Skips rows that already have populated fields
- **Duplicate Detection**: Won't re-import AP stars if already present
- **Batch Processing**: Gaia queries are batched for efficiency
- **Multiple Format Support**: Handles both letter-based (b, c, d) and numeric (0.01, 0.02) planet designations

## Requirements

```bash
pip install pandas numpy astropy astroquery
```

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
├── Aux_Targets.ipynb           # Main processing notebook
├── Pandora Snapshot Targets - *.csv  # Input CSV files
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
