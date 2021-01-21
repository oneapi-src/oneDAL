# Intel(R) oneAPI Data Analytics Library KDB\* Samples

KDB\* samples for the Intel(R) oneAPI Data Analytics Library (oneDAL) are designed to show how to use this library with a KDB database in a C++ application.

Unzip the archive with oneDAL samples to your working directory (`<sample_dir>`).

-----------------------------------------------------------------------------------------------------------------------------------------
**Note:** The Intel® oneAPI Data Analytics Library (oneDAL) is able to interface with kdb+. **Intel does not provide kdb+ or any license to kdb+ as part of oneDAL.** kdb+ is available for license from Kx Systems, Inc. under a commercial license (which may include fees) as well as a non-commercial license. You are solely responsible for procuring any license for kdb+ based on your use case.

## System Requirements
You can use oneDAL KDB samples on Linux\*, Windows\*, and macOS\* operating systems. For a list of oneDAL hardware and software requirements, refer to release notes for the version of oneDAL you are using.

### KDB implementations against which oneDAL has been validated:
- KDB+ 3.3

**Note:** oneDAL is expected to work on other KDB version as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### KDB support
You can download and install the KDB application from [the official web page][kdb]. Make sure to replace the `dataSourceName`, `dataSourcePort`, `dataSourceUsername` and `dataSourcePassword` names in the `datasource_kdb.cpp` sample file with the actual database address and credentials you plan to use:

Also, make sure to set the environment variables `KDB_HEADER_PATH` and `KDB_LIBRARY_PATH` with the correct paths before running the sample.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. oneDAL includes the `vars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `vars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
vars.bat {ia32|intel64}`
- On Linux\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source vars.sh {ia32|intel64}`
- On macOS\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source vars.sh`
For more information about setting environment variables and configuring oneDAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the `launcher` command with the `build` parameter:

```
cd <sample_dir>\cpp\kdb

launcher.bat {ia32|intel64} build
```

The command creates the `.\_results\ia32` or `.\_results\intel64` directory and builds `*.exe` executables and `*.exe` libraries, as well as creates a log file for build results.

To run oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the `launcher` command with the `run` parameter:

```
cd <sample_dir>\cpp\kdb

launcher.bat {ia32|intel64} run
```

Select the same architecture parameter as you provided to the launcher command with the build parameter.

For each sample, the results are placed into the `.\_results\ia32\<sample name>\.res` or `.\_results\intel64\<sample name>\.res` file, depending on the specified architecture.

### On Linux\*
To build oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/kdb

make {libia32|soia32|libintel64|sointel64} compiler={intel|gnu} mode=build
```

From the `{libia32|soia32|libintel64|sointel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the `make` command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/kdb

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

### On macOS\*
To build oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/kdb

make {libia32|dylibia32|libintel64|dylibintel64} compiler={intel|gnu|clang} mode=build
```

From the `{libia32|dylibia32|libintel64|dylibintel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `dylib` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `dylib`). For example: `_results/intel_intel64_a`.

To run oneDAL KDB C++ samples, go to the C++ KDB samples directory and execute the make command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/kdb

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

## Notices and Disclaimers
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017-2021, Intel Corporation

<!-- Links -->
[kdb]: https://kx.com/download
