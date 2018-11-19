# Intel(R) Data Analytics Acceleration Library Neural Networks Samples

Neural networks samples for the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) are designed to show how to use this library to create most common neural network topologies such as LeNet\*, AlexNet\*, GoogleNet\* in a C++ application.

Unzip the archive with Intel(R) DAAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use Intel(R) DAAL neural networks samples on Linux\*, Windows\*, and macOS\* operating systems. For a list of Intel(R) DAAL hardware and software requirements, refer to release notes for the version of Intel(R) DAAL you are using.

## Preparation Before Build and Run
### Preparing the Data Sets
Before running the sample, prepare the training and testing data sets.

## LeNet Sample

You can use this sample with the standard Mixed National Institute of Standards and Technology (MNIST) data sets. Use the following links to download the data sets:

[train-images-idx3-ubyte.gz][train-images-idx3] - training set images - 55000 training images and 5000 validation images

[train-labels-idx1-ubyte.gz][train-labels-idx1] - training set labels matching the images

[t10k-images-idx3-ubyte.gz][t10k-images-idx3] - test set images - 10000 images

[t10k-labels-idx1-ubyte.gz][t10k-labels-idx1] - test set labels matching the images

Before running the sample, download and decompress the data set into the following folder: `<sample_dir>\cpp\neural_networks\data`. Update the filenames if they do not match the names in the launcher.

## AlexNet Sample

For this sample the `<sample_dir>\cpp\neural_networks\data` directory already contains the **train_227x227.blob** and **test_227x227.blob** synthetic data sets. The **train_227x227.blob** data set contains five images for the AlexNet training stage, and the **test_227x227.blob** data set contains two images for the inference stage. The `*.blob` files data layout has following structure:

| Header, 16 bytes | Images, [*number_of_images x image_size*] bytes | Labels, [*number_of_images x 4*] bytes |
| :--------------- |:----------------------------------------------- | :--------------------------------------|
- The "Header" section consists of four unsigned 32-bit integer in the little-endian format:
    - Header[0] - number of images in the "Images" file section;
    - Header[1] - number of channels each image has;
    - Header[2] - image width;
    - Header[3] - image height.
- The "Images" section contains *number_of_images* images packed in the "layered" format (B, G, R). Each layer is packed sequentially and consists of [*image_width x image_height*] bytes (C unsigned chars in the little-endian format). Total number of bytes in "Images" section is equal [*number_of_images x number_of_channels x image_width x image_height*] bytes.
- The "Labels" section consists of *number_of_images* unsigned 32-bit integers in the little-endian format. Each label corresponds to the image with the same index.

## GoogleNet Sample

For this sample the `<sample_dir>\cpp\neural_networks\data` directory contains the **train_224x224.blob** and **test_224x224.blob** synthetic data sets with the data layout described in the previous section.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. Intel(R) DAAL includes the `daalvars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `daalvars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
daalvars.bat {ia32|intel64}`

- On Linux\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source daalvars.sh|csh {ia32|intel64}`

- On macOS\*, you can find the `daalvars.sh|daalvars.csh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source daalvars.sh|csh`
For more information about setting environment variables and configuring Intel(R) DAAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `launcher` script with the `build` parameter:

```
cd <sample_dir>\cpp\neural_networks

launcher.bat {ia32|intel64} build
```

The command creates the `.\_results\ia32` or `.\_results\intel64` directory and builds `.exe` executables and `.exe` libraries, as well as creates a log file for build results.

To run Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `launcher` script with the `run` parameter:

```
cd <sample_dir>\cpp\neural_networks

launcher.bat {ia32|intel64} run
```

Select the same architecture parameter as you provided to the `launcher` script with the `build` parameter.

For each sample, the results are placed into the `.\_results\ia32\<sample name>\.res` or `.\_results\intel64\<sample name>\.res` file, depending on the specified architecture.

### On Linux\*
To build Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/neural_networks

make {libia32|soia32|libintel64|sointel64} compiler={intel|gnu} mode=build
```

From the `{libia32|soia32|libintel64|sointel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `make` command in the run mode. For example, if you run the `daalvars` script with the `intel64` target:

```
cd <sample_dir>/cpp/neural_networks

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

### On macOS\*
To build Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/neural_networks

make {libia32|dylibia32|libintel64|dylibintel64} compiler={intel|gnu|clang} mode=build
```

From the `{libia32|dylibia32|libintel64|dylibintel64}` parameters, select the one that matches the architecture parameter you provided to the `daalvars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `dylib` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `dylib`). For example: `_results/intel_intel64_a`.

To run Intel(R) DAAL neural networks C++ samples, go to the C++ neural networks samples directory and execute the `make` command in the run mode. For example, if you run the `daalvars` script with the `intel64` target:

```
cd <sample_dir>/cpp/neural_networks

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
[train-images-idx3]: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
[train-labels-idx1]: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
[t10k-images-idx3]: http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
[t10k-labels-idx1]: http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
