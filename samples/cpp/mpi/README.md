# Intel(R) Data Analytics Acceleration Library Message-Passing Interface Samples

Message-passing interface (MPI) samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library with the Intel(R) MPI library in a C++ application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
The Intel(R) DAAL includes distributed algorithms that can run on MPI-based cluster environments of Linux\* and Windows\* operating systems with the Intel(R) MPI library. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

## Preparation Before Build and Run
### MPI support
To link an application with the Intel(R) MPI library, do the following:

- On Windows\*:
    1. Add the following string to the include path: `%ProgramFiles(x86)%\Intel\MPI\<ver>\intel64\include`, where `<ver>` is the directory for a particular MPI version;
    for example,`%ProgramFiles(x86)%\IntelSWTools\MPI\5.1.x.xxx\intel64\include`.
    2. Add the following string to the library path: `%ProgramFiles(x86)%\IntelSWTools\MPI\<ver>\intel64\lib`;
    for example, `%ProgramFiles(x86)%\IntelSWTools\MPI\5.1.x.xxx\intel64\lib`.
    3. Add `impi.lib` and `impicxx.lib` to your link command.
- On Linux\*, the Intel(R) MPI Library includes the `mpivars` scripts that you can run to set environment variables. You can find the `mpivars` scripts at `<MPI install-dir>/bin64/:
source mpivars.sh|csh`

Check the documentation that comes with your MPI implementation for implementation-specific details of linking.

### Setting Up the Build Environment 
Before you build the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `daalvars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
daalvars.bat intel64`
- On Linux OS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin\:
source daalvars.sh|csh intel64`

For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build Intel(R) DAAL MPI C++ samples, go to the C++ MPI samples directory and execute the `launcher` command with the `build` parameter:

```
cd <sample_dir>\cpp\mpi

launcher.bat build
```

The command creates the `.\_results\intel64` directory and builds `*.exe` executables and `*.exe` libraries, as well as creates a log file for build results.

To run Intel(R) DAAL MPI C++ samples, go to the C++ MPI samples directory and execute the `launcher` command with the `run` parameter:

```
cd <sample_dir>\cpp\mpi

launcher.bat run
```

For each sample, the results are placed into the `.\_results\intel64\<sample name>\.res` file.

### On Linux\*
To build Intel(R) DAAL MPI C++ samples, go to the C++ MPI samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/mpi

make {libintel64|sointel64} sample=<sample_name> mode=build
```

From the `{libintel64|sointel64}` parameters, select the one that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The names of the samples are available in the `daal.lst` file. If a sample name is not be provided, all samples are built.

The command creates a directory for the chosen library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL MPI C++ samples, go to the C++ MPI samples directory and execute the `make` command in the run mode. For example, if you run the `daalvars` script with the `intel64` target:

```
cd <sample_dir>/cpp/mpi 

make  libintel64 sample=svd_fast_distributed_mpi mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and the `svd_fast_distributed_mpi.cpp` sample and runs the executable.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804
