# daal4scripting - Simplified APIs to Intel(R) DAAL for R and Python

**_This is a technical preview, not a product. Intel might decide to discontinue this project at any time._**

With this API your R and Python programs can use Intel(R) DAAL algorithms in just one line:
```
kmeans_init(data, 10, t_method="plusPlusDense")
```
You can even run this on a cluster by simple adding a keyword-parameter
```
kmeans_init(data, 10, t_method="plusPlusDense", distributed=TRUE)
```

Please see GettingStartedHLDAAL.pdf for details.

# Building packages
## VARIABLES
The following variables are accepted at make time:
* PREFIX: directory to store packages
* BUILD_PREFIX: temporary build space
* SWIG: absolute path to swig
* DAAL4R_VERSION/DAAL4PY_VERSION: package version
## OVERVIEW
The build-process is 2-phased
1.	Creating sources from C++ headers and preparing for package build
    * Note: For your convenience the repository already contains the generated sources. So will usually not need this step.
2.	Building the binary package

The resulting package is a ready-to-install conda/R package. You can of course also directly install the package in phase 2.
## PREREQUISITES
### Prerequisites for building binary packages
Below is the list of dependences besides the obvious ones (like make, R, Python etc)
#### Linux
* A C++ compiler with C++11 support
* Intel(R) TBB (https://www.threadingbuildingblocks.org/)
* Intel(R) CnC version 1.2.00 or later (https://github.com/icnc/icnc) [R only]
* intel(R) DAAL version 2018 (https://github.com/01org/daal)
* conda build (https://github.com/conda/conda-build) [Python only]
#### Additional prerequisites on Windows
* Microsoft Visual Studio 2017
* Rtools (https://cran.r-project.org/bin/windows/Rtools/)

*Note: building python API is currently supported on Linux only.*
#### CnC
*Note: When using conda, the pre-built package from Intel's test channel on anaconda.org (-c intel/label/test) is available (either online or for download). You need to install/build CnC only if you do not use conda - like when building interfaces for R.*
```
CNC_VERSION=1.2.000
git clone https://github.com/icnc/icnc v$CNC_VERSION
python make_kit.py -r $CNC_VERSION –mpi=<root-of-intel-mpi-install>/intel64 --itac=NONE
```
This creates a ready-to-use CnC install at kit.pkg/cnc/1.2.000
For more details please visit https://github.com/icnc/icnc.
### Prerequisites for creating sources
* SWIG: A patched version is required, see https://github.com/swig/swig/pull/900)
* python and jinja2
## BUILDING BINARY PACKAGE DAAL4PY
*Note: the call to 'make touch' is required only if you want to re-use pre-built sources.*
### With conda
Requires DAALROOT to be set correctly.
```
cd python
make touch
make binpkg
```
You can also directly install the build by calling ```cd python; make touch; make install```.
### Without conda
Requires DAAL, TBB and CnC in your lib/include search path
```
cd python
make sources
cd dist/daal4py
python setup.py build
```
You can also directly install the build by replacing the last command with ```python setup.py install```.
## BUILDING BINARY PACKAGE DAAL4R
Requires DAALROOT, TBBROOT and CNCROOT to be set correctly.
```
cd r
make touch
make binpkg
```
You can also directly install the build by calling ```cd r; make touch; make install```.

*Note: On Windows it is recommended to work with administrator privileges as RTools/msys/mingw can cause issues if not.*
## BUILDING SOURCES
```
cd r # or cd python
make clean
make sources
```
