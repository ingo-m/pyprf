[![PyPI version](https://img.shields.io/pypi/v/pyprf.svg)](https://pypi.org/project/pyprf/)
[![tests](https://github.com/ingo-m/pyprf/actions/workflows/tests.yml/badge.svg)](https://github.com/ingo-m/pyprf/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ingo-m/pyprf/branch/main/graph/badge.svg)](https://codecov.io/gh/ingo-m/pyprf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.835161.svg)](https://doi.org/10.5281/zenodo.835161)

# PyPRF
<img src="https://raw.githubusercontent.com/ingo-m/pyprf/main/logo/logo.png" width=200 align="right" />

A free & open source *python package* for *population receptive field (pRF) analysis*. With this package you can present visual stimuli for a retinotopic mapping fMRI experiment, and perform a pRF analysis on the fMRI data.

PyPRF has two parts, which are installed and used separately:

| Part | What it does | Where it runs | What you need |
|------|--------------|---------------|---------------|
| [1. Stimulus presentation](#1-stimulus-presentation) | Presents the retinotopic mapping stimuli in the MRI scanner. | The stimulus computer of the MRI lab. | [PsychoPy](https://www.psychopy.org/) and a folder with two scripts. You do **not** need to install the `pyprf` package. |
| [2. Analysis](#2-analysis) | Estimates a pRF for each voxel, and creates retinotopic maps. | Your analysis computer. | `pip install pyprf` |

The stimuli consist of bars at different locations and orientations, filled with flickering black and white checkerboards. It is important that the participant fixates throughout the experiment. Therefore, there is a central fixation task: the fixation dot occasionally changes its colour, and the participant presses a button in response. At the end of the presentation, the participant's hit rate is provided as feedback.

A pRF is estimated for each voxel (see [1]). The pRF model is a 2D Gaussian; the free parameters are the Gaussian's x- and y-position, and its width (SD). This rather simple pRF model is best suited for early visual cortex (higher cortical areas may require more complex models).

## 1. Stimulus presentation

The stimulus presentation consists of two scripts that you open and run in PsychoPy. On purpose, they are not part of the `pyprf` python package: stimulus computers in MRI labs are often shared, and it is easiest to copy a folder and run the scripts from the PsychoPy app.

### Installation

1. Install [PsychoPy](https://www.psychopy.org/download.html) (the standalone installer is easiest). The stimulus presentation was tested with PsychoPy 2026.2.4.
2. Download `pyprf-stimulus-presentation-v<version>.zip` from the [releases page](https://github.com/ingo-m/pyprf/releases) (below "Assets"), copy it to the stimulus computer, and unzip it. (Alternatively, clone this repository and copy the folder `pyprf/stimulus_presentation`.)

The folder looks like this. Please do not rename or move the subfolders, the scripts find the design matrices and the log folder relative to their own location.

```
stimulus_presentation/
├── code/
│   ├── create_design_matrix.py   <- step 1: create a design matrix
│   └── stimulus.py               <- step 2: present the stimuli
├── design_matrices/              <- design matrices are saved here
│   ├── Run_01.npz                   (examples, see below)
│   ├── Run_02.npz
│   └── Run_03.npz
└── log/                          <- log files & stimulus logs are saved here
```

### Step 1: Create a design matrix

The design matrix specifies the experimental design: which stimulus is shown on each fMRI volume, and when the target events of the fixation task occur. There is one stimulus per fMRI volume, so the design matrix is made for a specific volume TR.

> **Note:** The example design matrices `Run_01` to `Run_03` were made for a TR of 2.079 s (227 volumes each). If your fMRI sequence has a different TR, you need to create your own design matrices. Because the order of the stimuli is randomised, you can create a different design matrix for each run of your experiment.

Open `code/create_design_matrix.py` in PsychoPy (Coder view) and click *Run* (or run `python create_design_matrix.py` in a terminal). Set the parameters in the dialog:

<img src="https://raw.githubusercontent.com/ingo-m/pyprf/main/logo/example_gui_design.png" width=400 />

- **Output file name**: By default, the first unused name (e.g. `Run_04`). If a design matrix with this name already exists, you can choose a different name, or overwrite the existing one.
- **TR [s]**: The volume TR of your fMRI sequence.
- **Number of bar orientations**: 4 (horizontal, vertical, and two diagonals), or 2 (horizontal & vertical).
- **Number of bar positions on x-axis / y-axis**: Number of bar positions along the width / height of the screen.
- **Number of blocks**: How often each combination of bar position & orientation is shown.
- **Number of rest trials**: Number of additional rest periods (3 volumes each) at random times.
- **Inter-trial interval for targets [s]**: Average time between target events (colour changes of the fixation dot).
- **Initial / final rest period [volumes]**: Rest (fixation only) at the beginning and end of the run.
- **Full screen**: If `True`, the bars cover the entire screen. If `False`, they cover a central square (with a side length equal to the screen height).
- **Stimulus contrasts**: One or two contrast levels of the bars.

After the design matrix is saved, a dialog shows its number of volumes and duration. Your fMRI sequence has to acquire (at least) this number of volumes.

### Step 2: Present the stimuli

Open `code/stimulus.py` in PsychoPy and click *Run* (or run `python stimulus.py` in a terminal).

<img src="https://raw.githubusercontent.com/ingo-m/pyprf/main/logo/example_gui_experiment.png" width=400 />

- **Design matrix**: Choose a design matrix. The list shows the TR and the number of volumes of each design matrix, please check that they match your fMRI sequence.
- **Logging mode**: `False` during the experiment (see step 3).
- **Scanner trigger key**: The key that the scanner sends as trigger pulse (see below).
- **Response key**: The key that the participant presses after a target event.
- **Width of monitor [cm]**, **Distance between observer and monitor [cm]**, **Width / Height of monitor [pixels]**: The properties of the screen in the scanner.

After you click *OK*, the fixation dot is shown, and the presentation waits for the trigger from the scanner. The presentation starts with the first trigger.

| Key | Default | |
|-----|---------|---|
| Scanner trigger key | `5` | Starts the stimulus presentation. The MRI scanner sends a key press at the start of each volume. Which key depends on your MRI lab (common are `5` and `t`). |
| Response key | `1` | The participant presses this key when the fixation dot changes its colour. |
| `e` + `x` | | Abort the presentation (keep `e` pressed, and press `x`). |

Keys are specified with their PsychoPy key names, for instance `5`, `t`, or `num_5` (number pad).

A log file of the presentation, including the participant's hits & misses, is saved in the `log` folder.

### Step 3: Create the stimulus log for the analysis

For the analysis, you need a log of the stimuli, i.e. an image of the visual stimulus on each fMRI volume. To create it, run `stimulus.py` again with **Logging mode** set to `True`, for each design matrix that you used, with the same screen settings as in the experiment. (This does not need a scanner trigger, and runs faster than the experiment. Don't use logging mode during an experiment.)

The stimulus log is saved as PNG images (one per volume) in `log/<design matrix>_frames/`, e.g. `log/Run_01_frames/frame_001.png`. Copy these folders to your analysis computer.

Please check that the bar stimuli are visible in the PNG images. (On some virtual or remote displays, the screen content cannot be captured, and the images are empty.)

## 2. Analysis

### Installation

Install `pyprf` from [PyPI](https://pypi.org/project/pyprf/), preferably in a [virtual environment](https://docs.python.org/3/library/venv.html):

```bash
pip install pyprf
```

`pyprf` requires Python 3.11 or newer. All dependencies are installed automatically.

### Configuration

The analysis parameters are set in a config file. Copy the [example config file](https://github.com/ingo-m/pyprf/blob/main/pyprf/analysis/config_default.csv) and adjust it to your data (see the comments in the file). The most important parameters are:

- `lstPathNiiFunc`: The fMRI data (nii files, one per run).
- `lstPathPng`: The stimulus logs from step 3 of the stimulus presentation (one per run, in the same order as the fMRI data), e.g. `['/my_data/Run_01_frames/frame_', '/my_data/Run_02_frames/frame_']`.
- `strPathNiiMask`: A brain mask (the pRF analysis is only performed within the mask).
- `varTr`: The volume TR.
- `varExtXmin`, `varExtXmax`, `varExtYmin`, `varExtYmax`: The extent of the stimulated area of the screen, in degrees of visual angle from the fixation point. In full screen mode, this is the entire screen; otherwise it is a central square with a side length equal to the screen height.
- `strVersion`: `'cython'` (recommended, considerably faster) or `'numpy'`.
- `varPar`: Number of processes to run in parallel.
- `strPathOut`: The basename of the output files.

On Windows, please use forward slashes in paths (e.g. `'C:/my_data/run_01.nii.gz'`).

### Run the analysis

```bash
pyprf -config /path/to/config.csv
```

The results are saved as nii files, e.g. `<strPathOut>_x_pos.nii.gz`, `_y_pos`, `_SD` (pRF size), `_R2`, `_polar_angle`, `_eccentricity`, and the parameter estimates `_PE_01` (one per stimulus contrast).

If the data do not fit into memory, you can switch on the hdf5 mode (`lgcHdf5 = True`). The data are then kept on disk instead of in memory, which is slower.

### Upgrading from pyprf 2

- `pyprf` 3 requires Python 3.11 or newer.
- The tensorflow (GPU) version of the analysis has been removed. Please set `strVersion = 'cython'` in your config file (if you used `strVersion = 'gpu'`).
- The analysis results are unchanged (this is checked by the tests).

## Contributions

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and run the tests.

## References

This application is based on the following work:

[1] Dumoulin, S. O. & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. NeuroImage 39, 647–660.

If you use `pyprf`, please cite it via its [Zenodo DOI](https://doi.org/10.5281/zenodo.835161).

## Support

Please use the [GitHub issues](https://github.com/ingo-m/pyprf/issues) for questions or bug reports.

## License

The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
