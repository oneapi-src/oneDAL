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

# Intel&reg; oneAPI Data Analytics Library

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Help](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/oneapi-src.oneDAL?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=5&branchName=master) ![License](https://img.shields.io/github/license/oneapi-src/oneDAL.svg)

Intel&reg; oneAPI Data Analytics Library (oneDAL) is a powerful machine learning library that helps speed up big data analysis. oneDAL solvers are also used in [Intel Distribution for Python](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) in Scikit-learn optimization.

Intel&reg; oneAPI Data Analytics Library is an extension of Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL).


## Table of Contents
- [Build yours high-performance data science application with intel&reg; oneDAL](#build-yours-high-performance-data-science-application-with-intel-onedal)
- [Python API](#python-api)
- [Scikit-learn patching](#scikit-learn-patching)
- [Distributed multi-node mode](#distributed-multi-node-mode)
- [oneDAL Apache Spark MLlib samples](#onedal-apache-spark-mllib-samples)
- [Installation](#installation)
- [Installation from Source](#installation-from-source)
- [Examples](#examples)
- [Samples](#samples)
- [Documentation](#documentation)
- [Technical Preview Features](#technical-preview-features)
- [oneDAL and Intel&reg; DAAL](#onedal-and-intel-daal)

## Build yours high-performance data science application with intel&reg; oneDAL

Intel&reg; oneDAL uses all capabilities of Intel&reg; hardware, which allows you to get an sugnificant performance boost on the classic machine learning algorithms.

We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

The size of the data is growing exponentially, as is the need for high-performance and scalable frameworks to analyze all this data and extract some benefits from it.
Besides superior performance on a single node, the oneDAL distributed computation mode also provides excellent strong and weak scaling (check charts below).

Intel&reg; oneDAL K-means fit, strong scaling result | Intel&reg; oneDAL K-means fit, weak scaling results
:-------------------------:|:-------------------------:
![](docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |   ![](docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

>*technical details: FPType: float32; HW: Intel Xeon Processor E5-2698 v3 @2.3GHz, 2 sockets, 16 cores per socket; SW: Intel® DAAL (2019.3), MPI4Py (3.0.0), Intel® Distribution Of Python (IDP) 3.6.8; Details available in the article https://arxiv.org/abs/1909.11822*

Check out our [examples](#examples)  and [documentation](#documentation)  for information about our API

## Python API

Intel&reg; oneDAL has a python API that is provided as a standalone python library called [daal4py](https://github.com/IntelPython/daal4py).
Below is an example of how daal4py can be used for calculation KMeans clusters

```python
import numpy as np
import pandas as pd
import daal4py as d4p

data = pd.read_csv("local_kmeans_data.csv", dtype = np.float32)

init_alg = d4p.kmeans_init(nClusters = 10,
                           fptype = "float",
                           method = "randomDense")

centroids = init_alg.compute(data).centroids
alg = d4p.kmeans(nClusters = 10, maxIterations = 50, fptype = "float",
                 accuracyThreshold = 0, assignFlag = False)
result = alg.compute(data, centroids)
```

### Scikit-learn patching

Python interface to efficient Intel® oneDAL provided by daal4py allows one to create scikit-learn compatible estimators, transformers, clusterers, etc. powered by oneDAL which are nearly as efficient as native programs.

| *Speedups of Intel&reg; oneDAL powered Scikit-learn over the original Scikit-learn, 28 cores, 1 thread/core* |
|:--:|
| ![](docs/readme-charts/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |
| *technical details: FPType: float32; HW: Intel(R) Xeon(R) Platinum 8276L CPU @ 2.20GHz, 2 sockets, 28 cores per socket; SW: scikit-learn 0.22.2, Intel® DAAL (2019.5), Intel® Distribution Of Python (IDP) 3.7.4; Details available in the article https://medium.com/intel-analytics-software/accelerate-your-scikit-learn-applications-a06cacf44912* |

daal4py have an API which matches API from scikit-learn.
This framework allows you to speed up your existing projects by changing one line of code

```python
from daal4py.sklearn.svm import SVC
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

svm = SVC(kernel='rbf', gamma='scale', C = 0.5).fit(X, y)
print(svm.score(X, y))
```

In addition daal4py provides an option to replace some scikit-learn methods by oneDAL solvers which makes it possible to get a performance gain **without any code changes**. This approach is the basis of Intel distribution for python scikit-learn. You can patch stock scikit-learn by using the only following commandline flag
```bash
python -m daal4py my_application.py
```
Patches can also be enabled programmatically:
```python
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from time import time

svm_sklearn = SVC(kernel="rbf", gamma="scale", C=0.5)

digits = load_digits()
X, y = digits.data, digits.target

start = time()
svm_sklearn = svm_sklearn.fit(X, y)
end = time()
print(end - start) # output: 0.141261...
print(svm_sklearn.score(X, y)) # output: 0.9905397885364496

from daal4py.sklearn import patch_sklearn
patch_sklearn() # <-- apply patch
from sklearn.svm import SVC

svm_d4p = SVC(kernel="rbf", gamma="scale", C=0.5)

start = time()
svm_d4p = svm_d4p.fit(X, y)
end = time()
print(end - start) # output: 0.032536...
print(svm_d4p.score(X, y)) # output: 0.9905397885364496
```

### Distributed multi-node mode

Often data scientists require different tools for analysis regular and big data. daal4py offers various processing models, which makes it easy to enable distributed multi-node mode.

```python
import numpy as np
import pandas as pd
import daal4py as d4p

d4p.daalinit() # <-- Initialize SPMD mode
data = pd.read_csv("local_kmeans_data.csv", dtype = np.float32)

init_alg = d4p.kmeans_init(nClusters = 10,
                           fptype = "float",
                           method = "randomDense",
                           distributed = True) # <-- change model to distributed

centroids = init_alg.compute(data).centroids

alg = d4p.kmeans(nClusters = 10, maxIterations = 50, fptype = "float",
                 accuracyThreshold = 0, assignFlag = False,
                 distributed = True)  # <-- change model to distributed

result = alg.compute(data, centroids)
```

For more details browse our [daal4py documentation](https://intelpython.github.io/daal4py/).

## oneDAL Apache Spark MLlib samples

<img align="right" style="display:inline;" height=300 width=550 src="docs/readme-charts/intel%20oneDAL%20Spark%20samples%20vs%20Apache%20Spark%20MLlib.png"></a>

oneDAL provides scala / java interfaces that match Apache Spark MlLib API and use oneDAL solvers under the hood. This implementation allows you to get a 3-18X increase in performance compared to default Apache Spark MLlib.

>*technical details: FPType: double; HW: 7 x m5.2xlarge AWS instances; SW: Intel DAAL 2020 Gold, Apache Spark 2.4.4, emr-5.27.0; Spark config num executors 12, executor cores 8, executor memory 19GB, task cpus 8*

Check [samples](#samples) tab for more details.

## Installation

You can install oneDAL:

- from [oneDAL home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) as a part of Intel&reg; oneAPI Base Toolkit.
- from [GitHub\*](https://github.com/oneapi-src/oneDAL/releases).

### Installation from Source
See [Installation from Sources](INSTALL.md) for details.

## Examples

Except C++ and Python API oneDAL also provide API for C++ SYCL and Java languages. Check out tabs below for more examples.
- [C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp)
- [oneAPI C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/cpp)
- [oneAPI DPC++](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/dpc)
- [Java](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/java)
- [Python](https://github.com/IntelPython/daal4py/tree/master/examples)

## Documentation
- [Get Started](http://oneapi-src.github.io/oneDAL/getstarted.html)
- [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)
- [oneDAL documentation](http://oneapi-src.github.io/oneDAL/)
- [Specification](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/oneapi-dal-release-notes.html)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

## Samples
Samples is an examples of how oneDAL can be used in different applications.
- [Apache Arrow](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/arrow)
- [KDB](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/kdb)
- [MPI](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mpi)
- [MySQL](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mysql)
- [oneCCL](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/oneccl)
- [Hadoop](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/hadoop)
- [Java Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/spark)
- [Scala Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/scala/spark)

## Technical Preview Features

Technical preview features are introduced to gain early feedback from developers. A technical preview feature is subject to change in the future releases. Using a technical preview feature in a production code base is therefore strongly discouraged.
In C++ APIs, technical preview features are located in `daal::preview` and `onedal::preview` namespaces. In Java APIs, technical preview features are located in packages that have the `com.intel.daal.preview` name prefix.
The preview features list:
- `MultiNodeBatch` for K-Means, a stepless distributed algorithm based on oneCCL
- Graph Analytics: 
	- Undirected graph without edge and vertex weights (undirected_adjacency_array_graph) - 32bit vertex index only
	- Jaccard Similarity Coefficients for all vertex pairs, a batch algorithm which processes the graph by blocks

## oneDAL and Intel&reg; DAAL

Intel&reg; oneAPI Data Analytics Library is an extension of Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL).

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL located under the `master` branch.

|Product|Latest release|Branch|Resources|
|-------|--------------|------|:-------------:|
|oneDAL       |2021.1-beta08|[master](https://github.com/oneapi-src/oneDAL)</br>[rls/onedal-beta08-rls](https://github.com/oneapi-src/oneDAL/tree/rls/onedal-beta08-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)|
|Intel&reg; DAAL|2020 Gold|[rls/daal-2020-u2-rls](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-u2-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/daal-programming-guide/top.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/intel-data-analytics-acceleration-library-2020-system-requirements.html)|
