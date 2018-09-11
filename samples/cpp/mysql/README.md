# Intel(R) Data Analytics Acceleration Library MySQL\* Samples

MySQL\* samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library with a MySQL database in a C++ application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL MySQL\* samples on Linux\*, Windows\*, and macOS\* operating systems. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### MySQL\* implementations against which Intel(R) DAAL has been validated:
- MySQL 5.6.22

**Note:** Intel(R) DAAL is expected to work on other MySQL version as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### MySQL support
You can download and install the MySQL\* application from [the official web page][mysql]. To be able to use MySQL\* C++  samples, make sure to configure the ODBC connector for the user who has permissions to create tables in the database. Also, make sure to replace the `mySQL_test` and `mySQL_test_32` database names in the `datasource_mysql.cpp` sample file with the actual database name you plan to use:

If your ODBC connector on Windows\* is installed in a directory different from `C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\`, make sure to update the `ODBC_PATH` variable in the `launcher.bat` script with the correct path before running the script.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. The Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `daalvars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
daalvars.bat {ia32|intel64}`
- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`
- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`

For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `launcher` command with the `build` parameter:

```
cd <sample_dir>\cpp\mysql

launcher.bat {ia32|intel64} build
```

The command creates the `.\_results\ia32` or `.\_results\intel64` directory and builds `*.exe` executables and `*.exe` libraries, as well as creates a log file for build results.

To run Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `launcher` command with the `run` parameter:

```
cd <sample_dir>\cpp\mysql

launcher.bat {ia32|intel64} run
```

Select the same architecture parameter as you provided to the `launcher` command with the `build` parameter.

For each sample, the results are placed into the `.\_results\ia32\<sample name>\.res` or `.\_results\intel64\<sample name>\.res` file, depending on the specified architecture.

### On Linux\*
To build Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/mysql

make {libia32|soia32|libintel64|sointel64} compiler={intel|gnu} mode=build
```

From the `{libia32|soia32|libintel64|sointel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command in the run mode. For example, if you run the `daalvars` script with the `intel64` target:

```
cd <sample_dir>/cpp/mysql

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

### On macOS\*
To build Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/mysql

make {libia32|dylibia32|libintel64|dylibintel64} compiler={intel|gnu|clang} mode=build
```

From the `{libia32|dylibia32|libintel64|dylibintel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `dylib` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `dylib`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command in the run mode. For example, if you run the `daalvars` script with the `intel64` target:

```
cd <sample_dir>/cpp/mysql

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804

<!-- Links -->
[mysql]: http://dev.mysql.com/downloads/
