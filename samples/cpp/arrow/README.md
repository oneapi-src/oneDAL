# Intel(R) Data Analytics Acceleration Library Apache Arrow\* Samples

Apache Arrow\* samples for the Intel® Data Analytics Acceleration Library (Intel® DAAL) are designed to show how to use this library with the Apache Arrow* library in a C++ application.

Unzip the archive with Intel® DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL Apache Arrow\* samples on Linux\*, Windows\*, and macOS\* operating systems. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### Apache Arrow\* implementations against which Intel(R) DAAL has been validated:
- Apache Arrow\* 0.15.1

**Note:** Intel(R) DAAL is expected to work on other Apache Arrow\* version as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### Apache Arrow\* support
You can download and install the Apache Arrow\* library from the [https://github.com/apache/arrow][arrow_repo]. Before running the sample make sure to set the environment variables `ARROWROOT` and `ARROWCONFIG` with the correct path and build type (debug or release, see [https://github.com/apache/arrow/blob/master/docs/source/developers/cpp.rst#building][arrow_building] web page for details). For Linux\* and MacOS\* only: set variables `BOOST_SYSTEM_LIBRARY_PATH` with the correct path to `libboost_system.a` and `BOOST_FILESYSTEM_LIBRARY_PATH` with the correct path to `libboost_filesystem.a`.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `vars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `vars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
vars.bat intel64`
- On Linux\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source vars.sh intel64`
- On macOS\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source vars.sh`
Pay attention that `32-bit` architectures are not supported by Apache Arrow\*, that's why we provide building this sample on the `64-bit` architectures only. For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the `launcher` command with the `build` parameter:

```
cd <sample_dir>\cpp\arrow

launcher.bat build
```

The command creates the `.\_results\intel64` directory and builds `*.exe` executables, as well as creates a log file for build results.

To run Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the `launcher` command with the `run` parameter:

```
cd <sample_dir>\cpp\arrow

launcher.bat run
```

Select the same architecture parameter as you provided to the launcher command with the build parameter.

For each sample, the results are placed into the `.\_results\intel64\<sample name>\.res` file.

### On Linux\*
To build Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/arrow

make {libintel64|sointel64} compiler={intel|gnu} mode=build
```

From the `{libintel64|sointel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the `make` command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/arrow

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

### On macOS\*
To build Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/arrow

make {libintel64|dylibintel64} compiler={intel|gnu} mode=build
```

From the `{libintel64|dylibintel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `dylib` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `dylib`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL Apache Arrow\* C++ samples, go to the C++ Apache Arrow\* samples directory and execute the make command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/arrow

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2019, Intel Corporation

#### Optimization Notice
>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804

<!-- Links -->
[arrow_repo]: https://github.com/apache/arrow
[arrow_building]: https://github.com/apache/arrow/blob/master/docs/source/developers/cpp.rst#building
