# Intel(R) Data Analytics Acceleration Library
![License](https://img.shields.io/github/license/intel/daal.svg)

Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes of computation.

## Transition to Open Development model
Development model for Intel(R) DAAL have been changed and now public github.com repository become main point for product development. Now we will have transparent commits history, public CI and public review process. You will see more changes going forward! 
Previous repository structure have bee cleaned and can be shared on request. Existing forks can be reused and  will require only pull-down for master branch. Details on branching schema will be updated in future.

- [How to contribute](#how-to-contribute)
- [System Requirements](#system-requirements)
- [Installation](#installation)
    - [Installation from Binaries](#installation-from-binaries)
    - [Installation from Sources](#installation-from-sources)

## License
Intel DAAL is licensed under Apache License 2.0.

## Online Release Notes and Documentation
You can find What's New features per release on [Intel(R) DAAL Release Notes](https://software.intel.com/en-us/articles/intel-daal-release-notes-and-new-features) web page and latest documentation on the [Intel(R) Data Analytics Acceleration Library Documentation](https://software.intel.com/en-us/intel-daal-support/documentation) web page.

## Deprecation Notice
With the introduction of [daal4py](https://intelpython.github.io/daal4py/index.html), a package that supersedes PyDAAL, Intel is deprecating PyDAAL and will discontinue support starting with Intel(R) DAAL 2021 and Intel(R) Distribution for Python 2021. Until then Intel will continue to provide compatible pyDAAL [pip](https://pypi.org/project/pydaal/) and [conda](https://anaconda.org/intel/pydaal) packages for newer releases of Intel DAAL and make it available in open source. However, Intel will not add the new features of Intel DAAL to pyDAAL. Intel recommends developers switch to and use [daal4py](https://github.com/IntelPython/daal4py).

## How to Contribute
We welcome community contributions to Intel DAAL. 
If you have an idea how to improve the product:

* Let us know about your proposal via [Issues on Intel(R) DAAL GitHub\*](https://github.com/intel/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library).

Or contribute your changes directly to repository through pull request:
* Make sure you can build the product and run all the examples with your patch.
* In case of a larger feature, provide a relevant example.
* [Submit](https://github.com/intel/daal/pulls) a pull request.

Public and private CIs are enabled for repository and should be passing for PR. We will review your contribution and, if any additional fixes or modifications are necessary, may give some feedback to guide you. When accepted, your pull request will be merged into GitHub* repository.

Intel DAAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## System Requirements
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

The lists below contain the system requirements necessary to support application development with Intel DAAL. We tested Intel DAAL on the operating systems and with the compilers listed below, but Intel DAAL is expected to work on many more Linux\* distributions as well.

Let us know if you have any troubles with the distribution you are using.

List of supported Operating Systems and tools may be found at [Intel DAAL web site](https://software.intel.com/en-us/articles/intel-daal-2019-system-requirements).

## Installation
You can install Intel DAAL from the provided binary packages or from the GitHub\* sources.

For platform-specific getting started documents, see the following pages:

* [Get Started with Intel(R) Data Analytics Acceleration Library for Windows*](https://software.intel.com/en-us/get-started-with-daal-for-windows)
* [Get Started with Intel(R) Data Analytics Acceleration Library for Linux*](https://software.intel.com/en-us/get-started-with-daal-for-linux)
* [Get Started with Intel(R) Data Analytics Acceleration Library for macOS*](https://software.intel.com/en-us/get-started-with-daal-for-osx)

### Installation from Binaries
You can download an archive from the [GitHub\* release page](https://github.com/intel/daal/releases). In the `daal/bin` directory of the downloaded archive you can find a script to set the environment variables for library usage.

If you have issues with running the script, you may need to replace the `INSTALLDIR` string in `daal/bin/daalvars.sh` and/or `daal/bin/daalvars.csh` with the name of the directory where you unpacked the archive.

### Installation from Sources

Required Software:
* C/C++ compiler (see [System Requirements](#system-requirements))
* Java\* JDK (see [System Requirements](#system-requirements))
* Microsoft Visual Studio\* (Windows\* only)
* [Git Large File Storage (LFS) extension](https://git-lfs.github.com)
* [MSYS2 installer](http://msys2.github.io) with the msys/make package (Windows\* only); install the package as follows:

        pacman -S msys/make

#### Installation Steps
1. Clone the sources from GitHub\* as follows:

        git clone --recursive https://github.com/intel/daal.git


2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only); for example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set an environment variables for one of the supported C/C++ compilers. For example:

    - **Microsoft Visual Studio\***:

            call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64

    - **Intel Compiler (Windows\*)**:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

    - **Intel Compiler (Linux\*)**:

            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

4. Set an environment variables for one of the supported Java\* compilers. For example:

    - **Windows\***:

            set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_77
            set PATH=%JAVA_HOME%\bin;%PATH%
            set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%

    - **macOS\***:

            export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home
            export PATH=$JAVA_HOME/bin:$PATH
            export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/darwin:$CPATH

    - **Linux\***:

            export JAVA_HOME=/usr/jdk/jdk1.6.0_02
            export PATH=$JAVA_HOME/bin:$PATH
            export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH

5. Download and set an environment for mklfpk libs:

    - **Windows\***:

            call .\scripts\mklfpk.bat

    - **Linux/macOS\***:

            scripts/mklfpk.sh [32|32e]

6. Download and install Intel(R) Threading Building Blocks (Intel(R) TBB) from https://github.com/intel/tbb:

    - **Windows\***:
        - Download and install free Community License Intel TBB. For more information, see [Get Intel(R) Performance Libraries for Free](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2).
        - Or build your own Intel TBB from https://github.com/intel/tbb
        - Set an environment variables for Intel TBB. For example:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\bin\tbbvars.bat" intel64 all

    - **Linux/macOS\***:
        - Use pre-build package or build TBB on your own. Or simply use scripts to dothis for you:

            scripts/tbb.sh [32|32e]

7. Build Intel DAAL via command-line interface. Choose the appropriate commands based on the platform and the compiler you use:

    - on **Linux\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=lnx32e

    - on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal PLAT=lnx32e COMPILER=gnu

    - on **macOS\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=mac32e

    - on **macOS\*** using **Clang\***:

            make -f makefile daal PLAT=mac32e COMPILER=clang

    - on **Windows\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=win32e

    - on **Windows\*** using **Microsoft Visual\* C++ Compiler**:

            make -f makefile daal PLAT=win32e COMPILER=vc

It is possible to build Intel DAAL libraries with selected set of algorithms and/or CPU optimizations. `CORE.ALGORITHMS.CUSTOM` and `REQCPUS` makefile defines are used for it.

- To build DAAL with Linear Regression and Support Vector Machine algorithms, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM="linear_regression svm" -j16


- To build DAAL with AVX2 and AVX CPU optimizations, run:

            make -f makefile daal PLAT=win32e REQCPU="avx2 avx" -j16


- To build DAAL with Moments of Low Order algorithm and AVX2 CPU optimizations, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM=low_order_moments REQCPU=avx2 -j16


Built libraries are located in the `\_\_release\_{os_name}/daal` directory.

## Python*
Intel DAAL can also be used with Python\* interfaces. You can find the daal4py package at  https://anaconda.org/intel/daal4py  See [PyDAAL Deprecation Notice](#deprecation-notice) for more information.

## See Also
* [Intel(R) DAAL Product Page](https://software.intel.com/en-us/intel-daal)
* [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)
