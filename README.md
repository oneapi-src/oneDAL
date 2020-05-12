<!--
******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

# Intel(R) oneAPI Data Analytics Library <!-- omit in toc -->

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Help](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/oneapi-src.oneDAL?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=5&branchName=master) ![License](https://img.shields.io/github/license/oneapi-src/oneDAL.svg)


Intel(R) oneAPI Data Analytics Library (oneDAL) is a library that helps speed up big data analysis. 
We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**. Our algorithms suppost **batch**, **online**, and **distributed** processing modes of computation. 

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

## Table of Contents <!-- omit in toc -->

- [oneDAL and Intel(R) DAAL](#onedal-and-intelr-daal)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [API](#api)


## oneDAL and Intel(R) DAAL

Intel(R) oneAPI Data Analytics Library is an extenstion of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL). 

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL located under the `master` branch.

|Product|Latest release|Branch|Resources|
|-------|--------------|------|:-------------:|
|oneDAL       |2021.1-beta06|[master](https://github.com/oneapi-src/oneDAL)</br>[rls/onedal-beta06-rls](https://github.com/oneapi-src/oneDAL/tree/rls/onedal-beta06-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/en-us/oneapi/onedal)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/en-us/articles/system-requirements-for-oneapi-data-analytics-library#)|
|Intel(R) DAAL|2020 Gold|[rls/daal-2020-rls](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-rls)</br>[rls/daal-2020-mnt](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-mnt) (contains ongoing fixes)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/en-us/daal)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Developer Guide](https://software.intel.com/en-us/daal-programming-guide)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/en-us/articles/intel-data-analytics-acceleration-library-2020-system-requirements)|

## Installation

You can install oneDAL: 

- from [oneDAL home page](https://software.intel.com/en-us/oneapi/onedal) as a part of Intel(R) oneAPI Base Toolkit.
- from [GitHub\*](https://github.com/oneapi-src/oneDAL/releases).

See [Installation from Sources](INSTALL.md) for details.

## Examples

Examples for different programming languages:

- [C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/cpp)
- [C++ SYCL*](https://github.com/oneapi-src/oneDAL/tree/master/examples/cpp_sycl)
- [Java](https://github.com/oneapi-src/oneDAL/tree/master/examples/java)

Data Examples for different computation modes:

- [Batch](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/batch)
- [Distributed](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/distributed)
- [Online](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/online)

## Documentation

- [Get Started](http://oneapi-src.github.io/oneDAL/getstarted.html)
- [System Requirements](https://software.intel.com/en-us/articles/system-requirements-for-oneapi-data-analytics-library#)
- [oneDAL documentation](http://oneapi-src.github.io/oneDAL/)
- [Specifications](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/en-us/articles/oneapi-dal-release-notes)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

## API

Intel(R) DAAL [provides](https://software.intel.com/en-us/articles/daal-api-reference) downloadable API References for C++, Python, and Java.

You can also use [daal4py](https://github.com/IntelPython/daal4py), a simplified Python API to Intel(R) DAAL that allows fast usage of the framework suited for Data Scientists or Machine Learning users.
