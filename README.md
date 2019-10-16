# Intel(R) Data Analytics Acceleration Library
[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/intel.daal?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=3&branchName=master) ![License](https://img.shields.io/github/license/intel/daal.svg)

Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes of computation.

## Transition to Open Development Model
The development model for Intel(R) DAAL has changed, and now the public GitHub* repository is where the product development takes place. From now on, we will have transparent commit history, public CI and public review process. You will see more changes going forward! 

We can share previous repository structure with you on request. Existing forks can be reused and only require pull-down for master branch. Details on branching schema will be updated in the future.

- [How to contribute](#how-to-contribute)
- [System Requirements](#system-requirements)
- [Installation](#installation)
    - [Installation from Binaries](#installation-from-binaries)
    - [Installation from Sources](#installation-from-sources)

## License
Intel(R) DAAL is licensed under Apache License 2.0.

## Online Release Notes and Documentation
See [Intel(R) DAAL Release Notes](https://software.intel.com/en-us/articles/intel-daal-release-notes-and-new-features) to find information on what's new in each release. Go to [Intel(R) Data Analytics Acceleration Library Documentation](https://software.intel.com/en-us/intel-daal-support/documentation) web page to find the latest documentation.

## Deprecation Notice
With the introduction of [daal4py](https://intelpython.github.io/daal4py/index.html), a package that supersedes PyDAAL, Intel is deprecating PyDAAL and will discontinue support starting with Intel(R) DAAL 2021 and Intel(R) Distribution for Python 2021. Until then Intel will continue to provide compatible pyDAAL [pip](https://pypi.org/project/pydaal/) and [conda](https://anaconda.org/intel/pydaal) packages for newer releases of Intel DAAL and make it available in open source. However, Intel will not add the new features of Intel DAAL to pyDAAL. Intel recommends developers switch to and use [daal4py](https://github.com/IntelPython/daal4py).

## How to Contribute
We welcome community contributions to Intel DAAL. If you have an idea how to improve the product, you can:

* Let us know about your proposal via [Issues on Intel(R) DAAL GitHub\*](https://github.com/intel/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library).
* Contribute your changes directly to the repository through [pull request](#pull-requests). 

### Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- For a larger feature, provide a relevant example.
- [Submit](https://github.com/intel/daal/pulls) a pull request.

Public and private CIs are enabled for Intel DAAL repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into GitHub* repository.

Intel DAAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## System Requirements
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

Go to the [Intel DAAL System Requirements](https://software.intel.com/en-us/articles/intel-daal-2019-system-requirements) page to find the list of supported operating systems and tools. The list contains system requirements necessary to support application development with Intel DAAL. We tested Intel DAAL on the operating systems and with the compilers listed there, but Intel DAAL is expected to work on many more Linux\* distributions as well.

Let us know if you have any problems with the distribution you are using.

## Installation
You can install Intel DAAL from the provided binary packages or from the GitHub\* sources.

For platform-specific get started guides, see the following pages:

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


2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

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

6. Download and install Intel(R) Threading Building Blocks (Intel(R) TBB):

    - **Windows\***:
        - Download and install free Community License Intel TBB (see [Get Intel(R) Performance Libraries for Free](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2)) or build your own Intel TBB from [Intel(R) TBB GitHub repository](https://github.com/intel/tbb).
        - Set environment variables for Intel TBB. For example:

                call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\bin\tbbvars.bat" intel64 all

    - **Linux/macOS\***:
        - Use pre-build package or build Intel TBB on your own. Alternatively, you can use scripts to do this for you:

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

It is possible to build Intel DAAL libraries with the pre-selected set of algorithms and/or CPU optimizations. To do this, use `CORE.ALGORITHMS.CUSTOM` and `REQCPU` flags defined in makefile. See examples below.

- To build DAAL with Linear Regression and Support Vector Machine algorithms, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM="linear_regression svm" -j16


- To build DAAL with AVX2 and AVX CPU optimizations, run:

            make -f makefile daal PLAT=win32e REQCPU="avx2 avx" -j16


- To build DAAL with Moments of Low Order algorithm and AVX2 CPU optimizations, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM=low_order_moments REQCPU=avx2 -j16

---
**NOTE:** Built libraries are located in the `__release_{os_name}/daal` directory.

---

## Python*
Intel DAAL can also be used with Python\* interfaces. Use [daal4py conda package](https://anaconda.org/intel/daal4py). See [PyDAAL Deprecation Notice](#deprecation-notice) for more information.

## See Also
* [Intel(R) DAAL Product Page](https://software.intel.com/en-us/intel-daal)
* [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)

## Legal Information

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document. Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

This document contains information on products, services and/or processes in development. All information provided here is subject to change without notice. Contact your Intel representative to obtain the latest forecast, schedule, specifications and roadmaps.

The products and services described may contain defects or errors known as errata which may cause deviations from published specifications. Current characterized errata are available on request.

Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

© 2019 Intel Corporation.

|Optimization Notice|
|:------------------|
|Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice. <br><br> Notice revision #20110804|
