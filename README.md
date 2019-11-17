<!--
******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/intel.daal?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=3&branchName=master) ![License](https://img.shields.io/github/license/intel/daal.svg)


Intel(R) oneAPI Data Analytics Library (oneDAL) is a library that helps speed up big data analysis 
by providing highly optimized algorithmic building blocks for all stages of data analytics 
(preprocessing, transformation, analysis, modeling, validation, and decision making) 
in batch, online, and distributed processing modes of computation. 
The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

## oneDAL and Intel(R) DAAL

Intel(R) oneAPI Data Analytics Library is an extenstion of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL). 

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL which is located under the master branch. If you want the gold quality library, use classical Intel(R) DAAL. 

Visit [oneDAL home page](https://software.intel.com/en-us/oneapi/onedal) for more information.

## Documentation

Browse [oneDAL documentation](http://intel.github.io/daal/) on GitHub or refer to the complete list of classical Intel(R) DAAL features and documentation are available at the official [Intel(R) DAAL website](https://software.intel.com/en-us/daal).

## Installation

You can install oneDAL: 

- from [oneDAL home page](https://software.intel.com/en-us/oneapi/onedal) as a part of Intel(R) Base Toolkit.
- from [GitHub\*](https://github.com/intel/daal/releases).

See [Installation from Sources](INSTALL.md) for details.

## How to Contribute
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. If you have an idea how to improve the product, you can:

* Let us know about your proposal via [Issues on oneDAL GitHub\*](https://github.com/intel/daal/issues).
* Contribute your changes directly to the repository through [pull request](#pull-requests). 

### Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- For a larger feature, provide a relevant example.
- [Submit](https://github.com/intel/daal/pulls) a pull request.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into GitHub* repository.

oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Python API

[daal4py](https://github.com/IntelPython/daal4py) is a simplified Python API to Intel(R) DAAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.

## See Also

* [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)

