# Intel(R) Data Analytics Acceleration Library
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes of computation.

## License
Intel DAAL is licensed under Apache License 2.0.

## Online Release Notes and Documentation
You can find What's New features per release on [Intel(R) DAAL Release Notes](https://software.intel.com/en-us/articles/intel-daal-release-notes-and-new-features) web page and latest documentation on the [Intel(R) Data Analytics Acceleration Library Documentation](https://software.intel.com/en-us/intel-daal-support/documentation) web page.

## How to Contribute
We welcome community contributions to Intel DAAL. If you have an idea how to improve the product:

* Let us know about your proposal via [https://github.com/intel/daal/issues](https://github.com/intel/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)
* Make sure you can build the product and run all the examples with your patch
* In case of a larger feature, provide a relevant example
* Submit a pull request at [https://github.com/intel/daal/pulls](https://github.com/intel/daal/pulls)

We will review your contribution and, if any additional fixes or modifications are necessary, may give some feedback to guide you. When accepted, your pull request will be merged into our internal and GitHub* repositories.

Intel DAAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## <a name="system-requirements">System Requirements</a>
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

The lists below contain the system requirements necessary to support application development with Intel DAAL. We tested Intel DAAL on the operating systems and with the compilers listed below, but Intel DAAL is expected to work on many more Linux* distributions as well.

Let us know if you have any troubles with the distribution you are using.

### List of supported Operating Systems and tools may be found at [Intel DAAL web site](https://software.intel.com/en-us/node/776616).

## Installation
You can install Intel DAAL from the provided binary packages or from the GitHub* sources.

For platform-specific getting started documents, see the following pages:

* [Getting Started with Intel(R) Data Analytics Acceleration Library for Windows*](https://software.intel.com/en-us/get-started-with-daal-for-windows)
* [Getting Started with Intel(R) Data Analytics Acceleration Library for Linux*](https://software.intel.com/en-us/get-started-with-daal-for-linux)
* [Getting Started with Intel(R) Data Analytics Acceleration Library for macOS*](https://software.intel.com/en-us/get-started-with-daal-for-osx)

### Installing from the Binaries
You can download an archive from the GitHub\* release page at [https://github.com/intel/daal/releases](https://github.com/intel/daal/releases). This archive contains a script to set the environment variables for library usage in the *daal/bin* directory.

If you have issues with running the script, you may need to replace the *INSTALLDIR* string in *daal/bin/daalvars.sh* and/or *daal/bin/daalvars.csh* with the name of the directory where you unpacked the archive.

### Installing from the Sources

#### Required Software
* C/C++ compiler (see [System Requirements](#system-requirements))
* Java\* JDK (see [System Requirements](#system-requirements))
* Microsoft Visual Studio\* (Windows* only)
* [Git Large File Storage (LFS) extension](https://git-lfs.github.com)
* [http://msys2.github.io](http://msys2.github.io) with the msys/make package (Windows* only); install the package as follows:

        pacman -S msys/make

#### Installation Steps
1. Clone the sources from GitHub* as follows:

        git clone --recursive https://github.com/intel/daal.git

2. Set the PATH environment variable to the MSYS2\* bin directory (Windows* only); for example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set an environment variables for one of the supported C/C++ compilers; for example:

 * Microsoft Visual Studio\*:

        call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64

 * Intel Compiler (Windows):

        call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

 * Intel Compiler (Linux):

        source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

4. Set an environment variables for one of the supported Java* compilers; for example:

 * Windows:

        set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_77
        set PATH=%JAVA_HOME%\bin;%PATH%
        set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%

 * macOS:

        export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home
        export PATH=$JAVA_HOME/bin:$PATH
        export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/darwin:$CPATH

 * Linux:

        export JAVA_HOME=/usr/jdk/jdk1.6.0_02
        export PATH=$JAVA_HOME/bin:$PATH
        export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH

5. Install Intel(R) Threading Building Blocks (Intel(R) TBB) (Windows* only)

    Download and install free Community License Intel TBB.
    See [this page](https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2) for more details.

    Set an environment variables for Intel TBB; for example:

        call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\bin\tbbvars.bat" intel64 all

6. Build Intel DAAL via the command-line interface with the following commands, depending on your platform:

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
