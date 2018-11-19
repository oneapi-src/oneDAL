# Intel(R) Data Analytics Acceleration Library Hadoop\* Samples

Hadoop\* samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library on the Hadoop cluster in a Java application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL Hadoop samples on Linux\* and macOS\* operating systems. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### Hadoop implementations against which Intel(R) DAAL has been validated:
- Hadoop 2.6.0

**Note:** Intel(R) DAAL is expected to work on many more Hadoop distributions as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### Setting Up the Build Environment 
Before you build the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`
- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`

For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Linux\* and macOS\*
To build Intel(R) DAAL Hadoop Java samples, go to the Java Hadoop samples directory:

```
cd <sample_dir>/java/hadoop
```

Execute the `./launcher.sh {ia32|intel64}` script to run the following algorithms on your Hadoop cluster:

- Correlation and variance-covariance matrices;
- Correlation and variance-covariance matrices using CSR numeric tables;
- K-Means clustering;
- K-Means clustering using CSR numeric tables;
- Linear regression using Normal Equations;
- Linear regression using QR decomposition-based method;
- Moments of low order matrices;
- Moments of low order matrices using CSR numeric tables;
- Naïve Bayes classifier;
- Naïve Bayes classifier using CSR numeric tables;
- Principal component analysis (PCA) using the singular value decomposition (SVD) method;
- Principal component analysis (PCA) using the correlation method;
- Principal component analysis (PCA) using the correlation method and CSR numeric tables;
- QR decomposition;
- Ridge regression using Normal Equations;
- Singular value decomposition (SVD).

You can manage the list of running samples by changing the `Hadoop_samples_list` variable in the `./launcher.sh` script.

From the `{ia32|intel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script. If no parameters are defined, the Intel(R) 64 architecture is used by default.

The command creates the `/Hadoop/<sample_name>` and `/Hadoop/Libraries` directories, builds and runs `<sample_name>.class` executables.

For each algorithm, the results are stored in the `/_results/<sample_name>/part-r-00000` sequence file.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804
