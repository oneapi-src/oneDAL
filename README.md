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

# Intel(R) oneAPI Data Analytics Library

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](http://intel.github.io/daal/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Help](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](#how-to-contribute)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/intel.daal?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=3&branchName=master) ![License](https://img.shields.io/github/license/intel/daal.svg)


Intel(R) oneAPI Data Analytics Library (oneDAL) is a library that helps speed up big data analysis. 
We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**. Our algorithms suppost **batch**, **online**, and **distributed** processing modes of computation. 

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

## oneDAL and Intel(R) DAAL

Intel(R) oneAPI Data Analytics Library is an extenstion of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL). 

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL located under the `master` branch. To learn about the difference between `releases` and `master` branches, read about [preview features](#preview-features).

|Product|Latest release|Branch|Resources|
|-------|--------------|------|:-------------:|
|oneDAL       |oneAPI Beta|[master](https://github.com/intel/daal)</br>[releases](https://github.com/intel/daal/tree/releases) (contains the latest stable version)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/en-us/oneapi/onedal)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Documentation](http://intel.github.io/daal/)&nbsp;&nbsp;&nbsp;|
|Intel(R) DAAL|2020 Gold|[rls/daal-2020-rls](https://github.com/intel/daal/tree/rls/daal-2020-rls)</br>[rls/daal-2020-mnt](https://github.com/intel/daal/tree/rls/daal-2020-mnt) (contains ongoing fixes)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/en-us/daal)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Developer Guide](https://software.intel.com/en-us/daal-programming-guide)&nbsp;&nbsp;&nbsp;|

## Table of Contents

- [Preview Features](#preview-features)
- [Installation](#installation)
- [Examples](#examples)
- [How to Contribute](#how-to-contribute)
- [Documentation](#documentation)
- [Python API](#python-api)

## Preview Features

Preview features are introduced to gain early feedback from developers. A preview feature is subject to change in the future releases. Using a preview feature in a production code base is therefore strongly discouraged.

- The `releases` branch contains the latest stable version of the library. The API for this branch can be considered stable.
- The `master` branch contains preview features along with the stable ones. The stability of API for this branch is not guaranteed.   

Users who wish to use, test, and provide feedback on the new features are encouraged to use the master branch. 

---
**Note:** The list of features that should be considered preview features is empty at this moment. When we add features with no guarantee of stability, features that reserve the right to change and break the API at any time, we will list them here.

---

## Installation

You can install oneDAL: 

- from [oneDAL home page](https://software.intel.com/en-us/oneapi/onedal) as a part of Intel(R) oneAPI Base Toolkit.
- from [GitHub\*](https://github.com/intel/daal/releases).

See [Installation from Sources](INSTALL.md) for details.

## Examples

Examples for different programming languages:

- [C++](https://github.com/intel/daal/tree/master/examples/cpp)
- [C++ SYCL*](https://github.com/intel/daal/tree/master/examples/cpp_sycl)
- [Java](https://github.com/intel/daal/tree/master/examples/java)

Data Examples for different computation modes:

- [Batch](https://github.com/intel/daal/tree/master/examples/data/batch)
- [Distributed](https://github.com/intel/daal/tree/master/examples/data/distributed)
- [Online](https://github.com/intel/daal/tree/master/examples/data/online)

## How to Contribute
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. If you have an idea how to improve the product, you can:

* Let us know about your proposal via [Issues on oneDAL GitHub\*](https://github.com/intel/daal/issues).
* Contribute your changes directly to the repository through [pull request](#pull-requests). 

### Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- Make sure your code is in line with our [coding style](#code-style).
- For a larger feature, provide a relevant example.
- [Submit](https://github.com/intel/daal/pulls) a pull request into the `master` branch.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

### Code Style

Our repository contains [clang-format configurations](https://github.com/intel/daal/blob/master/.clang-format) that you should use on your code. 
To do this, run:

```
clang-format style=file <your file>
```

Refer to [ClangFormat documentation](https://clang.llvm.org/docs/ClangFormat.html) for more information.

---
**Note:** oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---

## Documentation

- [Get Started](http://intel.github.io/daal/getstarted.html)
- [oneDAL documentation](http://intel.github.io/daal/)
- [Specifications](https://spec.oneapi.com/oneDAL/index.html)
- [Release Notes](https://software.intel.com/en-us/articles/oneapi-dal-release-notes)

## Python API

[daal4py](https://github.com/IntelPython/daal4py) is a simplified Python API to Intel(R) DAAL that allows fast usage of the framework suited for Data Scientists or Machine Learning users.
