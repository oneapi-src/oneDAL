# Intel(R) Data Analytics Acceleration Library
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes of computation.

## License
Intel DAAL is licensed under Apache License 2.0.

## Online Documentation
You can find the latest Intel DAAL documentation on the [Intel(R) Data Analytics Acceleration Library 2017 Documentation](https://software.intel.com/en-us/intel-daal-support/documentation) web page.

## How to Contribute
We welcome community contributions to Intel DAAL. If you have an idea how to improve the product:

* Let us know about your proposal via [https://github.com/01org/daal/issues](https://github.com/01org/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)
* Make sure you can build the product and run all the examples with your patch
* In case of a larger feature, provide a relevant example
* Submit a pull request at [https://github.com/01org/daal/pulls](https://github.com/01org/daal/pulls)

We will review your contribution and, if any additional fixes or modifications are necessary, may give some feedback to guide you. When accepted, your pull request will be merged into our internal and GitHub* repositories.

Intel DAAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## <a name="system-requirements">System Requirements</a>
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

The lists below contain the system requirements necessary to support application development with Intel DAAL. We tested Intel DAAL on the operating systems and with the compilers listed below, but Intel DAAL is expected to work on many more Linux* distributions as well.

Let us know if you have any troubles with the distribution you are using.

### Validated Operating Systems
* Windows* 8 (IA-32 / Intel(R) 64)
* Windows* 8.1 (IA-32 / Intel(R) 64)
* Windows* 10 (IA-32 / Intel(R) 64)
* Windows Server* 2008 R2 SP1 and SP2
* Windows HPC Server* 2008 R2
* Windows Server* 2012
* Red Hat Enterprise Linux* 6 (IA-32 / Intel(R) 64)
* Red Hat Enterprise Linux* 7 (IA-32 / Intel(R) 64)
* Red Hat Fedora Core* 20 (IA-32 / Intel(R) 64)
* Red Hat Fedora Core* 23 (IA-32 / Intel(R) 64)
* Red Hat Fedora Core* 24 (IA-32 / Intel(R) 64)
* SUSE Linux Enterprise Server* 11
* SUSE Linux Enterprise Server* 12
* Debian GNU/Linux* 8 (IA-32 / Intel(R) 64)
* Ubuntu* 14.04 LTS (IA-32 / Intel(R) 64)
* Ubuntu* 15.10 (IA-32 / Intel(R) 64)
* OS X\* 10.11 (Xcode* 7.0)
* macOS\* 10.12 (Xcode* 8.0)

### Validated C/C++ Compilers for Windows*
* Intel(R) C++ Compiler 16.0 for Windows* OS
* Intel(R) C++ Compiler 17.0 for Windows* OS
* Microsoft Visual Studio* 2013
* Microsoft Visual Studio* 2015

### Validated C/C++ Compilers for Linux*
* Intel(R) C++ Compiler 16.0 for Linux* OS
* Intel(R) C++ Compiler 17.0 for Linux* OS
* GNU Compiler Collection* 5.1 and later

### Validated C/C++ Compilers for macOS*
* Intel(R) C++ Compiler 16.0 for OS X\*
* Intel(R) C++ Compiler 17.0 for OS X\*
* Clang\* from Xcode* 7
* Clang\* from Xcode* 8

### Validated Java* Compilers:
* Java\* SE 8 from Sun Microsystems*

## Installation
You can install Intel DAAL from the provided binary packages or from the GitHub* sources.

For platform-specific getting started documents, see the following pages:

* [Getting Started with Intel(R) Data Analytics Acceleration Library for Windows*](https://software.intel.com/en-us/get-started-with-daal-for-windows)
* [Getting Started with Intel(R) Data Analytics Acceleration Library for Linux*](https://software.intel.com/en-us/get-started-with-daal-for-linux)
* [Getting Started with Intel(R) Data Analytics Acceleration Library for macOS*](https://software.intel.com/en-us/get-started-with-daal-for-osx)

### Installing from the Binaries
You can download an archive from the GitHub\* release page at [https://github.com/01org/daal/releases](https://github.com/01org/daal/releases). This archive contains a script to set the environment variables for library usage in the *daal/bin* directory.

If you have issues with running the script, you may need to replace the *INSTALLDIR* string in *daal/bin/daalvars.sh* and/or *daal/bin/daalvars.csh* with the name of the directory where you unpacked the archive.

### Installing from the Sources

#### Required Software
* C/C++ compiler (see [System Requirements](#system-requirements))
* Java\* JDK (see [System Requirements](#system-requirements))
* Microsoft Visual Studio\* (Windows* only)
* [http://msys2.github.io](http://msys2.github.io) with the msys/make package (Windows* only); install the package as follows:

        pacman -S msys/make

#### Installation Steps
1. Clone the sources from GitHub* as follows:

        git clone --recursive https://github.com/01org/daal.git

2. Set the PATH environment variable to the MSYS2\* bin directory (Windows* only); for example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set an environment variable for Microsoft Visual Studio\* (Windows* only); for example:

        call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64

4. Set an environment variable for one of the supported C/C++ compilers

5. Set an environment variable for one of the supported Java* compilers; for example:

        set PATH=C:\Program Files\Java\jdk1.8.0_77\bin;%PATH%
        set INCLUDE=C:\Program Files\Java\jdk1.8.0_77\include;C:\Program Files\Java\jdk1.8.0_77\include\win32;%INCLUDE%

6. Install Intel(R) Threading Building Blocks (Intel(R) TBB) (Windows* only)

    Download and install free Community License Intel TBB.
    See [this page](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2) for more details.

    Copy Intel TBB header files and libraries into Intel DAAL folder. E.g.:
        xcopy /I /Y /Q /E "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.2.187\windows\redist" %DAALDIR%\externals\tbb\win\redist
        xcopy /I /Y /Q /E "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.2.187\windows\tbb" %DAALDIR%\externals\tbb\win\tbb

7. Build Intel DAAL via the command-line interface with the following commands, depending on your platform:

 *  on Linux\* using Intel(R) C++ Compiler:

            make daal PLAT=lnx32e

 *  on Linux\* using GNU Compiler Collection\*:

            make daal PLAT=lnx32e COMPILER=gnu

 *  on macOS* using Intel(R) C++ Compiler:

            make daal PLAT=mac32e

 *  on macOS\* using Clang\*:

            make daal PLAT=mac32e COMPILER=clang

 *  on Windows* using Intel(R) C++ Compiler:

            make daal PLAT=win32e

 *  on Windows\* using Microsoft Visual* C++ Compiler:

            make daal PLAT=win32e COMPILER=vc

Built libraries are located in the *\_\_release\_{os_name}/daal* directory.

## Python*
Intel DAAL can be also used with Python\* interfaces. You can find the pyDAAL package at [http://anaconda.org/intel/pydaal](http://anaconda.org/intel/pydaal).

## See Also
* [Intel(R) DAAL Product Page](https://software.intel.com/en-us/intel-daal)
* [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)
