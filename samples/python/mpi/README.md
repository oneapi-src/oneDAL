# Intel(R) Data Analytics Acceleration Library Message-Passing Interface Samples

The Python\* Message-passing interface (MPI) samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library with the Intel(R) MPI library in a Python application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
Intel(R) DAAL includes distributed algorithms that can run on MPI-based cluster environments of Linux\* and Windows\* operating systems with the Intel(R) MPI library. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### Python MPI samples validated on:
- mpi4py 2.0.0 and Intel MPI 5.1

## Preparation Before Build and Run
### Setting Up the Environment
Before you run the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `daalvars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
daalvars.bat intel64`
- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin\:
source daalvars.sh|csh intel64`

For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

### Check Numpy Version
The minimum required version of numpy for use with PyDAAL samples is 1.9, and >= 1.10 is strongly recommended. Versions older than 1.9 may work, but have not been tested, so use at your own risk.

## Run Instructions
### On Linux\* and Windows\*
To run the Intel(R) DAAL MPI Python samples, go to the Python MPI samples directory and execute the `launcher.py` script. For example:

```
cd <sample_dir>/python/mpi

python launcher.py svd_fast_distributed_mpi
```

This command runs the `svd_fast_distributed_mpi.py` sample.

Running `launcher.py` with no arguments runs all the samples.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804
