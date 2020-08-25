

<!--
******************************************************************************
* Copyright 2014-2020 Intel Corporation
  *
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
  *
* http://www.apache.org/licenses/LICENSE-2.0
  *
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
  *******************************************************************************/-->

# Intel&reg; oneAPI Data Analytics Library

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](http://intel.github.io/daal/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Help](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/intel.daal?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=3&branchName=master) ![License](https://img.shields.io/github/license/intel/daal.svg)

 Intel&reg; oneAPI Data Analytics Library (oneDAL) is a powerful machine liearning library that helps speed up big data analysis.
 Intel&reg; oneDAL solvers are also used in [Intel Distribution for Python]([https://software.intel.com/en-us/distribution-for-python](https://software.intel.com/en-us/distribution-for-python)) in scikit-learn optimization.

## Build yours high-performance data science application with intel&reg; oneDAL

 intel&reg; oneDAL use all capabilities of your hardware, which allows you to get an incredible performance boost on the classic machine learning algorithms.
We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

Check out our [examples](#examples)  and [documentation](#documentation)  for information about our API

## Improve performance of your ML application

oneDAL is the perfect solution if you are interested in a high-performance machine learning framework. oneDAL uses machine learning algorithms optimizations and full power of your hardware to achieve greater performance than alternative solutions offers.

| *Speedups of Intel&reg; oneDAL powered Scikit-learn over the original Scikit-learn, 28 cores, 1 thread/core* |
|:--:|
| ![](https://github.com/oneapi-src/oneDAL/tree/master/docs/readme-charts/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |


The size of the data is growing exponentially, as is the need for high-performance and scalable frameworks to analyze all this data and extract some benefits from it.
Besides superior performance on a single node, the distribution mechanics of oneDAL provides excellent strong and weak scaling.


Intel&reg; oneDAL K-means fit, strong scaling result | Intel&reg; oneDAL K-means fit, weak scaling results
:-------------------------:|:-------------------------:
![](https://github.com/oneapi-src/oneDAL/tree/master/docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |
![](https://github.com/oneapi-src/oneDAL/tree/master/docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

## Installation

You can install oneDAL:

- from [oneDAL home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) as a part of Intel&reg; oneAPI Base Toolkit.
- from [GitHub\*](https://github.com/oneapi-src/oneDAL/releases).

## Installation from Source
See [Installation from Sources](INSTALL.md) for details.

## Technical Preview Features

Technical preview features are introduced to gain early feedback from developers. A technical preview feature is subject to change in the future releases. Using a technical preview feature in a production code base is therefore strongly discouraged.
In C++ APIs, technical preview features are located in `daal::preview` and `onedal::preview` namespaces. In Java APIs, technical preview features are located in packages that have the `com.intel.daal.preview` name prefix.
The only preview feature at the moment is `MultiNodeBatch` for K-Means, a stepless distributed algorithm based on oneCCL.

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

## Scikit-learn patching

Python interface to efficient Intel® Data Analytics and Acceleration Library (DAAL) provided by daal4py allows one to create scikit-learn compatible estimators, transformers, clusterers, etc. powered by DAAL which are nearly as efficient as native programs.
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


For more details browse our [daal4py documentation](https://intelpython.github.io/daal4py/).


## oneDAL Samples

<img align="right" style="display:inline;" height=300 width=550 src="https://github.com/oneapi-src/oneDAL/tree/master/docs/readme-charts/intel%20oneDAL%20Spark%20samples%20vs%20Apache%20Spark%20MLlib.png"></a>

Samples is a non official part of a project, that contains examples of how oneDAL can be used in different applications.

**oneDAL Spark Samples is a good example.**
oneDAL provides scala / java interfaces that match Apache Spark MlLib API and use oneDAL solvers under the hood. This implementation allows you to get a 3-18X increase in performance compared to default Apache Spark MLlib.

  ## Examples

Except C++ and Python API oneDAL also provide API for C++ SYCL and Java languages. Check out tabs below for more examples.
- [C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp)
- [C++ SYCL\*](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp_sycl)
- [Java](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/java)
- [Python](https://github.com/IntelPython/daal4py/tree/master/examples)

Data Examples for different computation modes:

- [Batch](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/data/batch)
- [Distributed](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/data/distributed)
- [Online](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/data/online)

## Documentation
- [Get Started](http://oneapi-src.github.io/oneDAL/getstarted.html)
- [oneDAL documentation](http://oneapi-src.github.io/oneDAL/)
- [Specifications](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/oneapi-dal-release-notes.html)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

## oneDAL and Intel&reg; DAAL

Intel&reg; oneAPI Data Analytics Library is an extenstion of Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL).

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL located under the `master` branch.

|Product|Latest release|Branch|Resources|
|-------|--------------|------|:-------------:|
|oneDAL       |2021.1-beta08|[master](https://github.com/oneapi-src/oneDAL)</br>[rls/onedal-beta08-rls](https://github.com/oneapi-src/oneDAL/tree/rls/onedal-beta08-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)|
|Intel&reg; DAAL|2020 Gold|[rls/daal-2020-rls](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-rls)</br>[rls/daal-2020-mnt](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-mnt) (contains ongoing fixes)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/daal-programming-guide/top.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/intel-data-analytics-acceleration-library-2020-system-requirements.html)|
