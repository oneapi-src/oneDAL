# Intel(R) Data Analytics Acceleration Library Neural Networks Samples

Neural networks samples included with the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library to create most common neural network topologies such as LeNet, GoogleNet, AlexNet, VGG-19, ResNet-50 in a Python application.

Unzip Intel(R) DAAL samples archive to your working directory (`<sample_dir>`)

## System Requirements
You can use Intel(R) DAAL neural networks samples on Linux\*, Windows\*, and macOS\* operating systems. For a detailed list of Intel(R) DAAL hardware and software requirements, refer to release notes of Intel(R) DAAL product you are using.

### Download DataSet
This sample can be used with standard MNIST dataset that can be downloaded following links below.

[train-images-idx3-ubyte.gz][train-images-idx3] - training set images - 55000 training images, 5000 validation images

[train-labels-idx1-ubyte.gz][train-labels-idx1] - training set labels matching the images

[t10k-images-idx3-ubyte.gz][t10k-images-idx3] - test set images - 10000 images

[t10k-labels-idx1-ubyte.gz][t10k-labels-idx1] - test set labels matching the images

Download and place uncompressed dataset in the following folder `<sample_dir>\python\neural_networks\data` prior to running this sample. Please change the file names if they don't match the names in the launcher.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. The Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables

- On Windows\*, you can find the `daalvars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
daalvars.bat {ia32|intel64}`
- On Linux OS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`
- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`

For more information about setting environment variables for different product suites, refer to product user guide

### Check Numpy Version
The minimum required version of numpy for use with PyDAAL samples is 1.9, and >= 1.10 is strongly recommended. Versions older than 1.9 may work, but have not been tested, so use at your own risk.

## Run Instructions
### On Windows\*, Linux\*, or macOS\*
To run Intel(R) DAAL neural networks Python samples, go to the Python neural networks samples directory and execute the `launcher` script:

```
cd <sample_dir>\python\neural_networks

python launcher.py
```

The command creates the `.\_results` directory and creates a log file for build results.

For each sample, the results are placed into the `.\_results\<sample name>\.res` file.

## Legal Information
Intel, and the Intel logo are trademarks of Intel Corporation in the U.S. and/or other countries.

\*Other names and brands may be claimed as the property of others.

&copy; Copyright 2017, Intel Corporation

#### Optimization Notice

>Intel's compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice.

>Notice revision \#20110804

<!-- Links -->
[train-images-idx3]: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
[train-labels-idx1]: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
[t10k-images-idx3]: http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
[t10k-labels-idx1]: http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
