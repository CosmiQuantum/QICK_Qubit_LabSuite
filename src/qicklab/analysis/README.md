## Todo: Discuss breaking these up into subfolders
## Naming Conventions

This directory contains all of our post-processing and plotting code for QickLab experiments. To keep it easy to navigate and maintain, please follow these rules when adding or renaming modules:

---

## 1. File names use **snake_case**

- Lowercase letters only.
- Words separated by underscores: `your_new_module.py`

## 2. Module Types & Patterns

- **Helper collections** (`*_tools.py`)  
  Group related functions that are often used together.  
  _Examples: `plot_tools.py`, `shot_tools.py`_

- **Foundational groups of functions** (`*_functions.py`)  
  Functions not tied to any particular context that are built on in other methods  
  _Example: `fit_functions.py`_

- **Data** (`*_data.py`)  
  Classes or functions for loading, parsing, and serializing data.  
  _Example: `h5_data.py`_

- **Spectroscopy analyses** (`*spec.py`)  
  Implements a specific spectroscopy workflow or algorithm.  
  _Examples: `qspec.py`, `resstarkspec.py`, `starkspec.py`_
