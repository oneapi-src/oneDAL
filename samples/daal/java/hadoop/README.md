# Intel(R) oneAPI Data Analytics Library Hadoop\* Samples

Hadoop\* samples for the Intel(R) oneAPI Data Analytics Library (oneDAL) are designed to show how to use this library on the Hadoop cluster in a Java application.

Unzip the archive with oneDAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use oneDAL Hadoop samples on Linux\* and macOS\* operating systems. For a list of oneDAL hardware and software requirements, refer to release notes for the version of oneDAL you are using.

### Hadoop implementations against which oneDAL has been validated:
- Hadoop 2.6.0

**Note:** oneDAL is expected to work on many more Hadoop distributions as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### Setting Up the Build Environment 
Before you build the sample, you must set certain environment variables that define the location of related libraries. oneDAL includes the `vars` scripts that you can run to set environment variables:

- On Linux\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source vars.s {ia32|intel64}`
- On macOS\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source vars.sh`

For more information about setting environment variables and configuring oneDAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Linux\* and macOS\*
To build oneDAL Hadoop Java samples, go to the Java Hadoop samples directory:

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

From the `{ia32|intel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script. If no parameters are defined, the Intel(R) 64 architecture is used by default.

The command creates the `/Hadoop/<sample_name>` and `/Hadoop/Libraries` directories, builds and runs `<sample_name>.class` executables.

For each algorithm, the results are stored in the `/_results/<sample_name>/part-r-00000` sequence file.

## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more at www.Intel.com/PerformanceIndex​.  

No product or component can be absolutely secure. 

Your costs and results may vary.

Intel technologies may require enabled hardware, software or service activation.

**&copy; Intel Corporation**. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others.

\*Other names and brands may be claimed as the property of others.
