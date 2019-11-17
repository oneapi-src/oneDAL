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
* C/C++ compiler (see [System Requirements](README.md#system-requirements))
* Java\* JDK (see [System Requirements](README.md#system-requirements))
* Microsoft Visual Studio\* (Windows\* only)
* [MSYS2 installer](http://msys2.github.io) with the msys/make package (Windows\* only); install the package as follows:

        pacman -S msys/make

## Installation Steps
1. Clone the sources from GitHub\* as follows:

        git clone --recursive https://github.com/intel/daal.git


2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set the environment variables for one of the supported C/C++ compilers. For example:

    - **Microsoft Visual Studio\***:

            call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64

    - **Intel Compiler (Windows\*)**:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

    - **Intel Compiler (Linux\*)**:

            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

4. Set the environment variables for one of the supported Java\* compilers. For example:

    - **Windows\***:

            set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_77
            set PATH=%JAVA_HOME%\bin;%PATH%
            set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%

    - **Linux\***:

            export JAVA_HOME=/usr/jdk/jdk1.6.0_02
            export PATH=$JAVA_HOME/bin:$PATH
            export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH

5. Download and set an environment for mklfpk libs:

    - **Windows\***:

            call .\scripts\mklfpk.bat

    - **Linux\***:

            scripts/mklfpk.sh [32|32e]

6. Download and install Intel(R) Threading Building Blocks (Intel(R) TBB):

    - **Windows\***:
        - Download and install free Community License Intel TBB (see [Get Intel(R) Performance Libraries for Free](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2)) or build your own Intel TBB from [Intel(R) TBB GitHub repository](https://github.com/intel/tbb).
        - Set the environment variables for Intel TBB. For example:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\bin\tbbvars.bat" intel64 all

    - **Linux\***:
        - Use pre-build package or build Intel TBB on your own. Alternatively, you can use scripts to do this for you:

            scripts/tbb.sh [32|32e]

7. Build oneDAL via command-line interface. Choose the appropriate commands based on the platform and the compiler you use:

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

