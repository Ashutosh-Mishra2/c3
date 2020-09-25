# c<sup>3</sup> - An integrated tool-set for Control, Calibration and Characterization

The c<sup>3</sup>po package is intended to close the loop between open-loop control optimization, control pulse calibration, and model-matching based on calibration data.

c<sup>3</sup>po  provides a simple Python API through which it may integrate with virtually any experimental setup.
Contact us at [c3@q-optimize.org](mailto://quantum.c3po@gmail.com).

Documentation is available [here](https://c3-toolset.readthedocs.io).

## Table of Contents
* [Downloading](#downloading)
* [Installation](#installation)  
* [Requirements](#requirements)  
* [Misc](#misc)  

<a name="downloading"><a/>
## Downloading
Until the project reaches the v1.0 release, source code will be provided upon request by the project team at Saarland University. Please contact us at [quantum.c3po@gmail.com](mailto://quantum.c3po@gmail.com).

<a name="installation"><a/>
## Installation (developer mode)

The easiest way to use the c<sup>3</sup>po package is through the installation with [pip](https://pypi.org/project/pip/), in developer mode.

Place the source files in the directory of your choice, and then run
```
pip install -e <c3po source directory>
```
Adding the -e specifies the developer option and will result in the source directory being linked into pip's index (and nothing will be downloaded, except any required dependencies, such as [QuTip](http://qutip.org/) and [pycma](https://github.com/CMA-ES/pycma)).

To update c<sup>3</sup>po at any point, simply update the files in the <c3po source directory>.


**Attention:** As explained above, this does only link the c<sup>3</sup>po folder to your
local python packages. Deleting the c<sup>3</sup>po folder does therefore also result in
the deletion of the c<sup>3</sup>po package.


<a name="requirements"><a/>
## Dependencies
- [QuTip](http://qutip.org/)
- [pycma](https://github.com/CMA-ES/pycma)
