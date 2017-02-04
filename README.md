# py_pRF_mapping (...work in progress...)
A free & open source *python tool for population receptive field analysis* consisting of two main parts:

### 1. Stimulus presentation  
A tool for presentation of visual stimuli during a retinotopic mapping fMRI experiment. The stimuli consist of bars at different locations and orientation, filled with flickering black and white chequerboards. There is a central fixation dot. It is important that the participant maintain fixation throughout the experiment. Therefore, we included a central fixation task. The fixation dot sometimes changes its colour, and the participant's task is to press a button (number '1') in response. At the end of the presentation, feedback is provided as to how many targets the participant detected.

### 2. Data analysis  
Analysis tools for fMRI data from retinotopic mapping experiment. A population receptive field (pRF) is estimated for each voxel (see [1]). The pRF model used here is a 2D Gaussian; the free parameters are the Gaussian's x- and y-position, and its width. This rather simple pRF model is best suited for early visual cortex (higher cortical areas may require more complex models).

## Dependencies
[**Python 2.7**](https://www.python.org/download/releases/2.7/)

| Stimulus presentation                    | Tested version |
|------------------------------------------|----------------|
| [Psychopy](http://www.Psychopy.org/)     | 1.83.04        |
| [NumPy](http://www.numpy.org/)           | 1.11.3         |
| [SciPy](http://www.scipy.org/)           | 0.18.1         |

| Data analysis                            | Tested version |
|------------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)           | 1.11.3         |
| [SciPy](http://www.scipy.org/)           | 0.18.1         |
| [NiBabel](http://nipy.org/nibabel/)      | 2.1.0          |
| [Cython](http://cython.org/) (optional¹) | 0.25.2         |

¹: For considerably faster performance

### Hardware specifications
...TODO...

## How to use

1. Clone or download the repository.

2. Stimulus presentation:

You can call the presentation script from command line:

``` bash
python ~/py_pRF_mapping/stimulus_presentation/Main/prfStim_Bars.py
```

Alternatively, you could start the Psychopy GUI and run the script form there (see [Psychopy documentation](http://www.Psychopy.org/documentation.html) for futher details).  
After starting the script you can enter *Participant ID* and *run number* in the general user interface (GUI). By default, the folder ```~/py_pRF_mapping/stimulus_presentation/Conditions/``` contains pseudo-randomised design matrices for 3 runs. In order to use these, enter '01', '02', or '03' in the respective field in the GUI. *If you would like to simply run the presentation one time, you can leave this setting at its default value ('01').*

After starting the script, it will wait for a trigger signal from the fMRI scanner (default: keyboard button number ```5```).

You can interrupt the presentation by pressing ```ESC```.

3. Data analysis:

In order to prepare the analysis, you need to run the stimulus presentation script with in *logging mode* in order to create a log of the stimulus presentation. Open ```~/py_pRF_mapping/stimulus_presentation/Main/prfStim_Bars.py``` in a text editor and set ```lgcLogMde = True```.

Now run the script either from command line or through the Psychoy GUI.

The stimulus presentation log is created in the folder ```~/py_pRF_mapping/stimulus_presentation/Log_<participant_ID>/pRF_mapping_log/Frames/```.

If you would like to use the cython functionality (much faster performance), you need to run the cython setup script first (in order to compile the cython functions). Change directory to the analysis folder:
```
cd ~/py_pRF_mapping/analysis
```

Compile the cython function:
```
python pRF_setup.py build_ext --inplace
```

Use the config file ```~/py_pRF_mapping/analysis/pRF_config.py``` in order to set up the analysis parameters. See comments therein for more information (TODO: more detailed documentation).

Run the analysis:
```
python ~/py_pRF_mapping/analysis/pRF_main.py
```

## References
This application is based on the following work:

[1] Dumoulin, S. O. & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. NeuroImage 39, 647–660.

## Support
Please use the [github issues](https://github.com/ingo-m/py_pRF_mapping/issues) for questions or bug reports.

## License
The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
