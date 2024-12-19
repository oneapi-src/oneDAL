<!--
******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# Installation from Sources

Required Software:
* C/C++ Compiler
* [DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) and [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) if building with SYCL support
* BLAS and LAPACK libraries - both provided by oneMKL
* Python version 3.9 or higher
* TBB library (repository contains script to download it)
* Microsoft Visual Studio\* (Windows\* only)
* [MSYS2](http://msys2.github.io) (Windows\* only)
* `make` and `dos2unix` tools; install these packages using MSYS2 on Windows\* as follows:

        pacman -S msys/make msys/dos2unix

For details, see [System Requirements for oneDAL](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/system-requirements-for-oneapi-data-analytics-library.html).

Note: the Intel(R) oneAPI components listed here can be installed together through the oneAPI Base Toolkit bundle:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

All of these dependencies can alternatively be installed through the `conda` software, but doing so will require a few additional setup steps - see [Conda Development Environment Setup](https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md#conda-development-environment-setup) for details.

## Docker Development Environment

[Docker file](https://github.com/uxlfoundation/oneDAL/tree/main/dev/docker) with the oneDAL development environment
is available as an alternative to the manual setup.

## Installation Steps


1. Clone the sources from GitHub\* as follows:

        git clone https://github.com/uxlfoundation/oneDAL.git

2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set the environment variables for one of the supported C/C++ compilers, such as [Intel(R)'s DPC compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html). For example:

    - **Microsoft Visual Studio\* 2022**:

            call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64

    - **Intel(R) C++ Compiler 19.1 (Windows\*)**:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

    - **Intel(R) C++ Compiler 19.1 (Linux\*)**:

            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Linux\*)**:

            source /opt/intel/oneapi/compiler/latest/env/vars.sh

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Windows\*)**:

            call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"

    Note: if the Intel compilers were installed as part of a bundle such as oneAPI Base Toolkit, it's also possible to set the environment variables at once for all oneAPI components used here (compilers, MKL, oneMKL, TBB) through the more general script that they provide - for Linux:

            source /opt/intel/oneapi/setvars.sh

4. Set up MKL:

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneMKL will already have been set up._

    Download and install [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).
    Set the environment variables for for Intel(R) oneMKL. For example:

    - **Windows\***:

            call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" intel64

    - **Linux\***:

            source /opt/intel/oneapi/mkl/latest/env/vars.sh

5. Set up Intel(R) Threading Building Blocks (Intel(R) TBB):

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneTBB will already have been set up._

    Download and install [Intel(R) TBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html).
    Set the environment variables for for Intel(R) TBB. For example:

    - oneTBB (Windows\*):

            call "C:\Program Files (x86)\Intel\oneAPI\tbb\latest\env\vars.bat" intel64

    - oneTBB (Linux\*):

            source /opt/intel/oneapi/tbb/latest/env/vars.sh intel64

    Alternatively, you can use scripts to do this for you (Linux\*):

            ./dev/download_tbb.sh

6. Download and install Python (version 3.9 or higher).

7. Build oneDAL via command-line interface. Choose the appropriate commands based on the interface, platform, and the compiler you use. Interface and platform are required arguments of makefile while others are optional. Below you can find the set of examples for building oneDAL. You may use a combination of them to get the desired build configuration:

    - DAAL interfaces on **Linux\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=lnx32e

    - DAAL interfaces on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal PLAT=lnx32e COMPILER=gnu

    - oneAPI C++/DPC++ interfaces on **Windows\*** using **Intel(R) DPC++ compiler**:

            make -f makefile oneapi PLAT=win32e

    - oneAPI C++ interfaces on **Windows\*** using **Microsoft Visual\* C++ Compiler**:

            make -f makefile oneapi_c PLAT=win32e COMPILER=vc

    - DAAL and oneAPI C++ interfaces on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal oneapi_c PLAT=lnx32e COMPILER=gnu

It is possible to build oneDAL libraries with selected set of algorithms and/or CPU optimizations. `CORE.ALGORITHMS.CUSTOM` and `REQCPUS` makefile defines are used for it.

- To build oneDAL with Linear Regression and Support Vector Machine algorithms, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM="linear_regression svm" -j16


- To build oneDAL with AVX2 and AVX512 CPU optimizations, run:

            make -f makefile daal PLAT=win32e REQCPU="avx2 avx512" -j16


- To build oneDAL with Moments of Low Order algorithm and AVX2 CPU optimizations, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM=low_order_moments REQCPU=avx2 -j16

On **Linux\*** it is possible to build debug version of oneDAL or the version that allows to do kernel profiling using <ittnotify.h>.

- To build debug version of oneDAL, run:

            make -f makefile daal oneapi_c PLAT=lnx32e REQDBG=yes

- To build oneDAL with kernel profiling information, run:

            make -f makefile daal oneapi_c PLAT=lnx32e REQPROFILE=yes

---
**NOTE:** Built libraries are located in the `__release_{os_name}[_{compiler_name}]/daal` directory.

---

After having built the library, if one wishes to use it for building [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex/tree/main) or for executing the usage examples, one can set the required environment variables to point to the generated build by sourcing the script that it creates under the `env` folder. The script will be located under `__release_{os_name}[_{compiler_name}]/daal/latest/env/vars.sh` and can be sourced with a POSIX-compliant shell such as `bash`, by executing something like the following from inside the `__release*` folder:

```shell
cd daal/latest
source env/vars.sh
```

The provided unit tests for the library can be executed through the Bazel system - see the [Bazel docs](https://github.com/uxlfoundation/oneDAL/tree/main/dev/bazel) for more information.

Examples of library usage will also be auto-generated as part of the build under path `daal/latest/examples/daal/cpp/source`. These can be built through CMake - assuming one starts from the release path `__release_{os_name}[_{compiler_name}]`, the following would do:

```shell
cd daal/latest/examples/daal/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

This will generate executables under path `daal/latest/examples/daal/cpp/_cmake_results/{platform_name}`. They can be executed as follows (note that they require access to the data files under `daal/latest/examples/daal/data`), assuming that one starts from inside the `build` folder (as at the end of the previous step):

```shell
cd ..
./_cmake_results/{platform_name}/{example}
```

For example, in a Linux platform, assuming one wishes to execute the `adaboost_dense_batch` example:

```shell
./_cmake_results/intel_intel64_so/adaboost_dense_batch
```

## Conda Development Environment Setup

The previous instructions assumed system-wide installs of the necessary dependencies. These can also be installed at a user-level through the `conda` or [mamba](https://github.com/conda-forge/miniforge) ecosystems.

First, create a conda environment for building oneDAL, after `conda` has been installed:

```shell
conda create -y -n onedal_env
conda activate onedal_env
```

Then, install the necessary dependencies from the appropriate channels with `conda`:

* **Linux\***:

```shell
conda install -y \
    -c https://software.repos.intel.com/python/conda/ \ `# Intel's repository`
    -c conda-forge \ `# conda-forge, for tools like 'make'`
    make python>=3.9 \ `# used by the build system`
    dpcpp-cpp-rt dpcpp_linux-64 intel-sycl-rt \ `# Intel compiler packages`
    tbb tbb-devel \ `# required TBB packages`
    mkl mkl-devel mkl-static mkl-dpcpp mkl-devel-dpcpp \ `# required MKL packages`
    cmake `# required to build the examples only`
```

* **Windows\***:

```bat
conda install -y^
    -c https://software.repos.intel.com/python/conda/^
    -c conda-forge^
    make dos2unix python>=3.9^
    dpcpp-cpp-rt dpcpp_win-64 intel-sycl-rt^
    tbb tbb-devel^
    mkl mkl-devel mkl-static mkl-dpcpp mkl-devel-dpcpp^
    cmake
```

Then modify the relevant environment variables to point to the conda-installed libraries:

* **Linux\***:

```shell
export MKLROOT=${CONDA_PREFIX}
export TBBROOT=${CONDA_PREFIX}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
export CPATH="${CONDA_PREFIX}/include:${CPATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib/cmake:${CMAKE_PREFIX_PATH}"
```

* **Windows\***:

```bat
set MKLROOT=%CONDA_PREFIX%\Library
set TBBROOT=%CONDA_PREFIX%\Library
set "LD_LIBRARY_PATH=%CONDA_PREFIX%\Library\lib;%LD_LIBRARY_PATH%"
set "LIBRARY_PATH=%CONDA_PREFIX%\Library\lib;%LIBRARY_PATH%"
set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
set "PKG_CONFIG_PATH=%CONDA_PREFIX%\Library\lib\pkgconfig;%PKG_CONFIG_PATH%"
set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library\lib\cmake;%CMAKE_PREFIX_PATH%"
```

After that, it should be possible to build oneDAL and run the examples using the ICX compiler and the oneMKL libraries as per the instructions.

For other setups in **Linux\***, such as building for platforms like `aarch64` that are not supported by Intel's toolkits or using non-default options offered by the Makefile, other software can be installed as follows:

* GCC compilers (option `COMPILER=gnu`):

```shell
conda install -y -c conda-forge \
    gcc gxx c-compiler cxx-compiler
```

(no environment variables are needed for `COMPILER=gnu`)

* Reference (non-tuned) computational backends, and BLAS/LAPACK backends from OpenBLAS (both through option `BACKEND_CONFIG=ref`):

```shell
conda install -y -c conda-forge \
    blas=*=*openblas* openblas
```

* Optionally, if one wishes to install the OpenMP variant of OpenBLAS instead of the pthreads one, or to use the ILP64 variant:
```shell
conda install -y -c conda-forge \
    blas=*=*openblas* openblas-ilp64=*=openmp*
```

Then set environment variables as needed:
```shell
export OPENBLASROOT=${CONDA_PREFIX}
```

(note that other variables such as `TBBROOT` and `CMAKE_PREFIX_PATH` are still required)
