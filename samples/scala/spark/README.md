# Intel(R) Data Analytics Acceleration Library Spark\* Samples

Spark\* samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library on the Spark cluster in a Scala application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL Spark samples on Linux\* and macOS\* operating systems. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### Spark implementations against which Intel(R) DAAL has been validated:
- Spark 2.0.0
### Scala samples validated on:
- Scala 2.11.11

**Note:** Intel(R) DAAL is expected to work on many more Spark distributions as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### Setting Up the Build Environment 
Before you build the sample, you must set certain environment variables that define the location of related libraries. The Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`
- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`

For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Linux\* and macOS\*
To build Intel(R) DAAL Spark Scala samples, go to the Scala Spark samples directory:

```
cd <sample_dir>/scala/spark
```

Execute the `./launcher.sh {ia32|intel64}` script to run the following algorithms on your Spark cluster:

- K-Means clustering;
- Principal component analysis (PCA) using the correlation method;

You can manage the list of running samples by changing the `Spark_samples_list` variable in the `./launcher.sh` script.

From the `{ia32|intel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script. If no parameters are defined for the `make` command, the Intel(R) 64 architecture is used by default.

For each algorithm, the results are stored in the `/_results/<sample_name>/<sample_name>.res` file.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804
