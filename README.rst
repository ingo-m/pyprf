`PyPI version <https://badge.fury.io/py/pyprf>`__ `Build
Status <https://travis-ci.org/ingo-m/pyprf>`__
`codecov <https://codecov.io/gh/ingo-m/pyprf>`__
`DOI <https://doi.org/10.5281/zenodo.1421970>`__

PyPRF
=====

A free & open source *python package* for *population receptive field
(pRF) analysis*. With this package you can present visual stimuli for a
retinotopic mapping fMRI experiment, and perform a pRF analysis on the
fMRI data.

1. Stimulus presentation
~~~~~~~~~~~~~~~~~~~~~~~~

Presents visual stimuli for retinotopic mapping experiments. The stimuli
consist of bars at different locations and orientations, filled with
flickering black and white checkerboards. It is important that the
participant fixates throughout the experiment. Therefore, there is a
central fixation task. The fixation dot occasionally changes its colour,
and the task is to press a button (number ``1``) in response. At the end
of the presentation, the participant’s hit rate is provided as feedback.

2. Data analysis
~~~~~~~~~~~~~~~~

Analysis tools for fMRI data from retinotopic mapping experiment. A pRF
is estimated for each voxel (see [1]). The pRF model used here is a 2D
Gaussian; the free parameters are the Gaussian’s x- and y-position, and
its width (SD). This rather simple pRF model is best suited for early
visual cortex (higher cortical areas may require more complex models).

How to use - stimulus presentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Installation

The stimulus presentation is implemented in
`Psychopy <http://psychopy.org/>`__, so if you would like to run the
experiment, you first need to install it (if you already have data and
just would like to run the analysis, you can skip this step). On
`debian <https://www.debian.org/>`__, Psychopy can easily be installed
using ``apt-get``:

.. code:: bash

   sudo apt-get install psychopy

If you’re running some other operating system, please refer to the
`Psychopy website <http://psychopy.org/>`__.

For the stimulus presentation, you do not need to install ``pyprf``. You
only need a copy of the folder ``~/pyprf/pyprf/stimulus_presentation``
The easiest way to get the ``pyprf`` stimuli is to clone the github
repository:

.. code:: bash

   git clone https://github.com/ingo-m/pyprf.git

(Or click the download button.)

Then you can simply copy the folder ``stimulus_presentation`` and all
its contents to the computer that you use for stimulus presentation. (Do
not change the folder names. )

2. Create design matrix

Before you can run the experiment, you need to create a design matrix in
which you specify the experimental design (e.g. how many repetitions of
the stimulus, inter trial interval for target events, fMRI volume TR,
etc.). You can either open the script
``~/pyprf/pyprf/stimulus_presentation/code/create_design_matrix.py`` in
Psychopy and run it from there, or call it directly at command line.

You can specify all parameters in the GUI that will pop up. Note that
there is one stimulus per fMRI volume, so you have to know the volume TR
when creating the design matrix.

3. Stimulus presentation

In order to present the stimuli, you can open the file
``~/pyprf/pyprf/stimulus_presentation/code/stimulus.py`` in Psychopy and
run it from there. Alternatively, you can call the presentation script
directly from command line:

.. code:: bash

   python `~/pyprf/pyprf/stimulus_presentation/code/stimulus.py`

A GUI will open where you can specify further experimental parameters.
Importantly, the name of the design matrix (e.g. ‘Run_01’) needs to
match that of the file you created in the previous step.

After starting the script, it will wait for a trigger signal from the
fMRI scanner (default: keyboard button ``5``).

The stimuli look like this:

You can interrupt the presentation by pressing ``e`` and ``x`` at the
same time.

How to use - analysis
~~~~~~~~~~~~~~~~~~~~~

1. Install ``numpy``. For instance:

.. code:: bash

   pip install numpy

(Or, alternatively, if you’re using conda,
``conda install -c conda-forge numpy``.)

2. The ``pyprf`` package can directly be installed from PyPI, in the
   following way:

.. code:: bash

   pip install pyprf

(Alternatively, you could also installed it from the repository, like
this: ``git clone https://github.com/ingo-m/pyprf.git`` followed by
``pip install /path/to/pyprf``.)

3. Data analysis:

In order to prepare the analysis, you need to run the stimulus
presentation script in *logging mode* in order to create a log of the
stimulus presentation. Run
``~/pyprf/pyprf/stimulus_presentation/code/stimulus.py`` (as described
above, either from Psychopy or at command line). In the GUI, set
‘Logging mode’ to ``True``.

The stimulus presentation log is created in the folder
``~/pyprf/pyprf/stimulus_presentation/log/Run_*_frames/``.

The analysis parameters are set in a config file. An example file can be
found at ``~/pyprf/pyprf/analysis/config_default.csv``. See comments
therein for more information.

Run the analysis:

.. code:: bash

   pyprf -config /path/to/config.csv

Dependencies
~~~~~~~~~~~~

``pyprf`` is implemented in `Python 3.6 <https://www.python.org/>`__.

If you install ``pyprf`` using ``pip`` (as described above), all of the
following dependencies except for ``Psychopy`` and ``numpy`` are
installed automatically - you do not have to take care of this yourself.
Simply follow the above installation instructions.

+--------------------------------------------------------+----------------+
| Stimulus presentation                                  | Tested version |
+========================================================+================+
| `Psychopy <http://www.Psychopy.org/>`__                | 1.83.04        |
+--------------------------------------------------------+----------------+
| `NumPy <http://www.numpy.org/>`__                      | 1.15.1         |
+--------------------------------------------------------+----------------+
| `SciPy <http://www.scipy.org/>`__                      | 1.1.0          |
+--------------------------------------------------------+----------------+
| `Pillow <https://pypi.python.org/pypi/Pillow/4.3.0>`__ | 5.0.0          |
+--------------------------------------------------------+----------------+

+--------------------------------------------------------+----------------+
| Data analysis                                          | Tested version |
+========================================================+================+
| `NumPy <http://www.numpy.org/>`__                      | 1.15.1         |
+--------------------------------------------------------+----------------+
| `SciPy <http://www.scipy.org/>`__                      | 1.1.0          |
+--------------------------------------------------------+----------------+
| `NiBabel <http://nipy.org/nibabel/>`__                 | 2.2.1          |
+--------------------------------------------------------+----------------+
| `Cython <http://cython.org/>`__\ ¹                     | 0.27.1         |
+--------------------------------------------------------+----------------+
| `Pillow <https://pypi.python.org/pypi/Pillow/4.3.0>`__ | 5.0.0          |
+--------------------------------------------------------+----------------+
| `Tensorflow <https://www.tensorflow.org/>`__\ ²        | 1.4.0          |
+--------------------------------------------------------+----------------+

¹: For considerably faster performance

²: Can yield fast performance, depending on hardware. However, requires
tensorflow to be configured for GPU usage (additional tensorflow
specific dependencies, including GPU drivers).

The analysis can be carried out in three different ways: using
`numpy <http://www.numpy.org/>`__, `cython <http://cython.org/>`__, or
`tensorflow <https://www.tensorflow.org/>`__. You can set this option in
the ``config.csv`` file. All three approaches yield the same results,
but differ in their dependencies and computational time: - **Numpy**
uses numpy for the model fitting. Should work out of the box. -
**Cython** offers a considerable speedup by using compiled cython code
for model fitting. Should work out of the box. *This approach is
recommended for most users*. - **Tensorflow** may outperform the other
options in terms of speed (depending on the available hardware) by
running the GLM model fitting on the graphics processing unit (GPU).
However, in order for this to work, tensorflow needs to be configured to
use the GPU (including respective drivers). See the
`tensorflow <https://www.tensorflow.org/>`__ website for information on
how to configure your system to use the GPU. If you do not configure
tensorflow to use the GPU, the analysis should still run without error
on the CPU. Because this analysis may run single-threaded, it would be
slow. Numpy is always required, no matter which option you choose.

Contributions
~~~~~~~~~~~~~

For contributors, we suggest the following procedure:

-  Create your own fork (in the web interface, or by
   ``git checkout -b new_branch``)

   -  If you create the branch in the web interface, pull changes to
      your local repository (``git pull``)

-  Change to new branch: ``git checkout new_branch``
-  Make changes
-  Commit changes to new branch (``git add .`` and ``git commit -m``)
-  Push changes to new branch (``git push origin new_branch``)
-  Create a pull request using the web interface

References
~~~~~~~~~~

This application is based on the following work:

[1] Dumoulin, S. O. & Wandell, B. A. (2008). Population receptive field
estimates in human visual cortex. NeuroImage 39, 647–660.

Support
~~~~~~~

Please use the `github
issues <https://github.com/ingo-m/pyprf/issues>`__ for questions or bug
reports. You can also contact us on the ``pyprf``
`gitter <https://gitter.im/pyprf/Lobby>`__ channel.

License
~~~~~~~

The project is licensed under `GNU General Public License Version
3 <http://www.gnu.org/licenses/gpl.html>`__.
