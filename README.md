# QICK_Qubit_LabSuite

The qubit analysis and data acquisition codebase for the CosmiQuantum group at Fermilab using the [QICK](https://github.com/openquantumhardware/qick) open source qubit controller. Specifically for qubit operations at the NEXUS and QUIET cryogenic facilities.

Custodians:
- Olivia Seidel (UT Arlington)
- Joyce Christiansen-Salameh (Cornell)
- Dylan Temples (FNAL)

## Table of Contents

- [Codebase Structure](#codebase-structure)
- [Conventions & Requirements](#conventions--requirements)
- [Usage](#usage)
- [Contributing](#contributing)
  
## Codebase Structure
The codebase is organized as follows:
- `configs/`
    - `NEXUS/`
    - `QUIET/`
- `scripts/`
- `src/qicklab/`
    - `analysis/`
    - `config/`
    - `experiments/`
    - `hardware/`
    - `utils/`
- `tests/`

### Conventions & Requirements
Each of these directories has specific conventions and requirements, as outlined below.

#### `configs/`
The actual configuration files used to run data acquisition live here. Each facility has its own directory to contain configuration files relevant for collecting qubit data there.

#### `scripts/`
Here, one can place high-level analysis scripts that `import` the `qicklab` package. These should be true scripts, and not Python files that are collections of helper methods (*i.e.,* these files should not be `import`ed). These scripts can be examples of how to do analysis, DAQ scripts, or actual analyses that we will be running repeatedly.

**Requirements.** The body of the script should be inside the `if __name__ == __main__:` conditional as is Pythonic standard.

**Conventions.** It is helpful to use `argparse` to accept command-line arguments to the script, so that the contents do not need to be edited regularly.

#### `src/qicklab`
The "meat" of the codebase. This is the backend, where all methods that should be `import`ed for analysis or acquisition live. No true scripts live here. Each subdirectory here has its own requirements and conventions, below.

##### `src/qicklab/analysis`
The mid-level code for analyzing qubit data. These methods can be used for offline post-analysis, or integrated with acquisition scripts for online analysis.

**Requirements.** These must have **no dependencies** on `qick`. These files can freely depend on other files within this directory or in `src/qicklab/utils`. 

**Conventions.** File names should be indicative of what types of methods are contained in the file, and the methods should be logically grouped. Each specific analysis type should be a class in its own file. See `auto_threshold.py`, `qspec.py`, `resstarkspec.py`, `ssf.py`, `starkspec.py`, and `t1.py` for examples.

##### `src/qicklab/config`
These are the files that handle reading config files and setting the configuration for data acquisition.

##### `src/qicklab/experiments`
These are the files that define the data acquisition (and online analysis) for any specific experiment we may wish to run. These will naturally **have dependency on `qick`**.

##### `src/qicklab/hardware`
These are the files that directly interface with the QICK board or the `qick` package. Running any of the experiments will depend on these files.

##### `src/qicklab/utils`
These are low-level helper methods that are of general use/interest. 
- `ana_utils`: Utility functions for analysis. These should not care about the specific structure of the data, but are generally useful for *e.g.*, sorting data, performing rolling averages, etc.
- `data_utils`: Utility functions for handling data types. Again, these do not care about specific data structures, but are methods for *e.g.*, converting between various data types and strings, sorting lists, and other data-handling tasks.
- `file_utils`: Utility functions for creating directories, finding files, and parsing files. The latter is dependent on the specific data structure, so in a future release, these are likely to move into classes for each data structure.
- `log_utils`: Utility functions for logging execution info to a file.
- `time_utils`: Utility functions to handle timestamps, datetimes, and other time-based manipulations.
- `visdom_utils:` Utility functions to interface with the visdom process.

**Requirements.** These must have **no dependencies** on files elsewhere in `src/qicklab`. They can, however, have dependencies to other files within `src/qicklab/utils` (just be mindful of circular imports, which will cause errors). 

**Conventions.** File names should be indicative of what types of methods are contained in the file, and the methods should be logically grouped.


#### `tests/`
TBD... likely unit tests and test acquisition methods. These can be run to ensure that changes to the codebase do not change the expected output of various acquisition and analysis methods. The files here are subject to the same requirements and conventions as in `scripts/`

## Usage

To install this package navigate to this directory and run this in the command line:
```
pip install .
```
If it is already installed and you need to update the package, run:
```
pip install --upgrade .
```
Or, just re-run the `pip install .` command. Note that these commands need to be run from the directory `/path/to/QICK_Qubit_LabSuite`.

### Offline Analysis
For offline analysis scripts, your preamble should include the lines
```
from qicklab.analysis import *
from qicklab.utils import *
```
(Or, better yet, replacing `*` with only the methods and classes you plan to use in your analysis.) This avoids having any dependencies on QICK, and will allow you to perform offline analysis of previously collected data on any machine.

### Data Acquisition
For data acquisition scripts, just go ahead and import the whole shebang:
```
import qicklab
```

## Contributing
If you'd like to add a new analysis, follow these steps.
1. Create a new branch using the instructions found at the CosmiQ [How to use github](https://github.com/CosmiQuantum/how_to_use_github) repo.
2. Create a Python file containing a class that defines the analysis you want to do. It should include pulling the data from disk and parsing it into useful data structures. Please use the following naming convention:
   - If your file contains a class, give the file the same name as your class using CamelCase.
   - If your file does not contain a class, please use the naming convention all_lowercase_with_underscores.py
   - For methods, please use the convention all_lowercase_with_underscores_definition()
   - If a file has a class, it should only contain that class in it
4. Move that file into `src/qicklab/analysis`.
5. Open the file `src/qicklab/analysis/__init__.py` and add a line importing your class:
```
from .myAnaClass import myAnaClass
```
And that's it! Now when you `from qicklab.analysis import *`, your analysis will be ready for use.


If you'd like to add a new measurement/experiment, follow these steps...
