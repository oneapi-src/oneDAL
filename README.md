﻿<!--
******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# oneAPI Data Analytics Library <!-- omit in toc --> <img align="right" width="100" height="100" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg">

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](#documentation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](#support)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/oneapi-src.oneDAL?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=5&branchName=master) [![License](https://img.shields.io/github/license/oneapi-src/oneDAL.svg)](https://github.com/oneapi-src/oneDAL/blob/master/LICENSE) [![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/oneapi-src/oneDAL/discussions)

oneAPI Data Analytics Library (oneDAL) is a powerful machine learning library that helps you accelerate big data analysis at all stages: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

The library implements classical machine learning algorithms. The boost in their performance is achieved by leveraging the capabilities of Intel&reg; hardware.

oneDAL is part of [oneAPI](https://oneapi.io). The current branch implements version 1.1 of oneAPI Specification.

## Usage

There are different ways for you to build high-performance data science applications that use the advantages of oneDAL:
- Use oneDAL C++ interfaces with or without SYCL support ([learn more](https://oneapi-src.github.io/oneDAL/#oneapi-vs-daal-interfaces))
- Use [Intel(R) Extension for Scikit-learn*](https://intel.github.io/scikit-learn-intelex/) to accelerate existing scikit-learn code without changing it
- Use [daal4py](https://github.com/intel/scikit-learn-intelex/tree/master/daal4py), a standalone package with Python API for oneDAL
Deprecation Notice: The Java interfaces are deprecated in the oneDAL library and may no longer be supported in future releases.


## Installation

Check [System Requirements](https://oneapi-src.github.io/oneDAL/system-requirements.html) before installing oneDAL.

You can [download the specific version of oneDAL](https://github.com/oneapi-src/oneDAL/releases) or [install it from sources](INSTALL.md).

## Examples

C++ Examples:

- [oneAPI interfaces with SYCL support](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/dpc)
- [oneAPI interfaces without SYCL support](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/cpp)
- [DAAL interfaces](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp)
  
Python Examples:
- [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks)
- [daal4py](https://github.com/intel/scikit-learn-intelex/tree/master/examples/daal4py)

<details><summary>Other Examples</summary>

- [MPI](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mpi)
- [MySQL](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mysql)
Deprecation Notice: The Java interfaces are deprecated in the oneDAL library and may no longer be supported in future releases. This includes Scala, Spark and Hadoop samples


</details>

## Documentation

oneDAL documentation:

- [Release Notes](https://github.com/oneapi-src/oneDAL/releases) 
- [Get Started Guide](https://oneapi-src.github.io/oneDAL/quick-start.html)
- [Developer Guide and Reference](http://oneapi-src.github.io/oneDAL/)

Other related documentation:

- [daal4py documentation](https://intelpython.github.io/daal4py/)
- [Intel(R) Extension for Scikit-learn* documentation](https://intel.github.io/scikit-learn-intelex/)
- [oneDAL Specifications](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)

## Apache Spark MLlib

oneDAL library is used for Spark MLlib acceleration as part of [OAP MLlib](https://github.com/oap-project/oap-mllib) project and allows you to get a **3-18x** increase in performance compared to the default Apache Spark MLlib.

<img style="display:inline;" height=300 width=550 src="docs/readme-charts/intel%20oneDAL%20Spark%20samples%20vs%20Apache%20Spark%20MLlib.png"></a>

>*Technical details: FPType: double; HW: 7 x m5.2xlarge AWS instances; SW: Intel DAAL 2020 Gold, Apache Spark 2.4.4, emr-5.27.0; Spark config num executors 12, executor cores 8, executor memory 19GB, task cpus 8*

## Scaling

oneDAL supports distributed computation mode that shows excellent results for strong and weak scaling:

oneDAL K-Means fit, strong scaling result | oneDAL K-Means fit, weak scaling results
:-------------------------:|:-------------------------:
![](docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |   ![](docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

>*Technical details: FPType: float32; HW: Intel Xeon Processor E5-2698 v3 @2.3GHz, 2 sockets, 16 cores per socket; SW: Intel® DAAL (2019.3), MPI4Py (3.0.0), Intel® Distribution Of Python (IDP) 3.6.8; Details available in the article https://arxiv.org/abs/1909.11822*

## Support

Ask questions and engage in discussions with oneDAL developers, contributers, and other users through the following channels:

- [GitHub Discussions](https://github.com/oneapi-src/oneDAL/discussions)
- [Community Forum](https://community.intel.com/t5/Intel-oneAPI-Data-Analytics/bd-p/oneapi-data-analytics-library)

You may reach out to project maintainers privately at onedal.maintainers@intel.com.

### Security <!-- omit in toc -->

To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

### Contribute <!-- omit in toc -->

We welcome community contributions. Check our [contributing guidelines](CONTRIBUTING.md) to learn more.

## License <!-- omit in toc -->

oneDAL is distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more information.

[oneMKL FPK microlibs](https://github.com/oneapi-src/oneDAL/releases/tag/Dependencies)
are distributed under Intel Simplified Software License.
Refer to [third-party-programs-mkl.txt](third-party-programs-mkl.txt) for details.
