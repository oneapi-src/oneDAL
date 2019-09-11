# Intel(R) Data Analytics Acceleration Library Spark\* Samples

Spark\* samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library on the Spark cluster in a Python\* application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL Spark samples on Linux\* and macOS\* operating systems. For a list of Intel DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

### Spark implementations against which Intel(R) DAAL has been validated:
- Spark 1.1.1
### PySpark samples validated on:
- PySpark 1.6.1

**Note:** Intel(R) DAAL is expected to work on many more Spark distributions as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`
- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`
### Check Numpy Version
The minimum required version of numpy for use with PyDAAL samples is 1.9, and >= 1.10 is strongly recommended. Versions older than 1.9 may work, but have not been tested, so use at your own risk.

## Build and Run Instructions
### On Linux\* and macOS\*
To run Intel(R) DAAL Spark Python samples, go to the Python Spark samples directory:

```
cd <sample_dir>/python/spark
```

Execute the `./launcher.sh {ia32|intel64}` script to run the following algorithms on your Spark cluster:

- Correlation and variance-covariance matrices;
- Correlation and variance-covariance matrices using CSR numeric tables;
- Implicit alternating least squares (ALS) using CSR numeric tables;
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
- Principal component analysis (PCA) using the correlation method and using CSR numeric tables;
- QR decomposition;
- Singular value decomposition (SVD).

You can manage the list of running samples by changing the Spark_samples_list variable in the `./launcher.sh` script.

From the `{ia32|intel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script. If no parameters are defined for the make command, the Intel(R) 64 architecture is used by default.

For each algorithm, the results are stored in the `/_results/<sample_name>/<sample_name>.res` file.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804
