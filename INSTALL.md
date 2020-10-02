<!--
******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
* Java\* JDK
* Microsoft Visual Studio\* (Windows\* only)
* [MSYS2 installer](http://msys2.github.io) with the msys/make package (Windows\* only); install the package as follows:

        pacman -S msys/make

For details, see [System Requirements for oneDAL](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html).

## Installation Steps
1. Clone the sources from GitHub\* as follows:

        git clone https://github.com/oneapi-src/oneDAL.git


2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set the environment variables for one of the supported C/C++ compilers. For example:

    - **Microsoft Visual Studio\* 2019**:

            call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" amd64

    - **Intel(R) C++ Compiler 19.1 (Windows\*)**:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

    - **Intel(R) C++ Compiler 19.1 (Linux\*)**:

            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

    - **Intel(R) oneAPI DPC++/C++ Compiler 2021.1 (Windows\*)**:

            call "C:\Program Files (x86)\inteloneapi\compiler\latest\env\vars.bat" intel64

4. Set the environment variables for one of the supported Java\* compilers. For example:

    - **Windows\***:

            set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_77
            set PATH=%JAVA_HOME%\bin;%PATH%
            set INCLUDE=%JAVA_HOME%\include;%INCLUDE%

    - **Linux\***:

            export JAVA_HOME=/usr/jdk/jdk1.6.0_02
            export PATH=$JAVA_HOME/bin:$PATH
            export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH

5. Download and set an environment for micromkl libs:

    - **Windows\***:

            .\dev\download_micromkl.bat

    - **Linux\***:

            ./dev/download/download_micromkl.sh

6. Download and install Intel(R) Threading Building Blocks (Intel(R) TBB):

    Download and install free Community License Intel(R) TBB (see [Get Intel(R) Performance Libraries for Free](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2)).
    Set the environment variables for for Intel(R) TBB. For example:

    - oneTBB (Windows\*):

            call "C:\Program Files (x86)\inteloneapi\tbb\latest\env\vars.bat" intel64

    - Intel(R) TBB 2020 (Linux\*):

            source /opt/intel/compilers_and_libraries_2020.0.166/linux/tbb/bin/tbbvars.sh intel64

    Alternatively, you can use scripts to do this for you (Linux\*):

            ./dev/download_tbb.sh

7. Download and install Python 3.7 (Windows\* only).

8. Build oneDAL via command-line interface. Choose the appropriate commands based on the platform and the compiler you use:

    - on **Linux\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=lnx32e

    - on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal PLAT=lnx32e COMPILER=gnu

    - on **Windows\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=win32e

    - on **Windows\*** using **Microsoft Visual\* C++ Compiler**:

            make -f makefile daal PLAT=win32e COMPILER=vc

It is possible to build oneDAL libraries with selected set of algorithms and/or CPU optimizations. `CORE.ALGORITHMS.CUSTOM` and `REQCPUS` makefile defines are used for it.

- To build oneDAL with Linear Regression and Support Vector Machine algorithms, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM="linear_regression svm" -j16


- To build oneDAL with AVX2 and AVX CPU optimizations, run:

            make -f makefile daal PLAT=win32e REQCPU="avx2 avx" -j16


- To build oneDAL with Moments of Low Order algorithm and AVX2 CPU optimizations, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM=low_order_moments REQCPU=avx2 -j16



---
**NOTE:** Built libraries are located in the `__release_{os_name}/daal` directory.

---

