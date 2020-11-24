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

<<<<<<< HEAD
# Intel(R) oneAPI Data Analytics Library <!-- omit in toc -->
=======
# Intel&reg; oneAPI Data Analytics Library <!-- omit in toc -->
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](#documentation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Samples](#samples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Get Help](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/oneapi-src.oneDAL?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=5&branchName=master) ![License](https://img.shields.io/github/license/oneapi-src/oneDAL.svg)


Intel(R) oneAPI Data Analytics Library (oneDAL) is a library that helps speed up big data analysis. 
We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**. Our algorithms suppost **batch**, **online**, and **distributed** processing modes of computation. 

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

## Table of Contents <!-- omit in toc -->

<<<<<<< HEAD
- [Technical Preview Features](#preview-features)
- [oneDAL and Intel(R) DAAL](#onedal-and-intelr-daal)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [API](#api)
=======
## Table of Contents <!-- omit in toc -->
- [Python API](#python-api)
  - [Scikit-learn patching](#scikit-learn-patching)
  - [Distributed multi-node mode](#distributed-multi-node-mode)
- [oneDAL Apache Spark MLlib samples](#onedal-apache-spark-mllib-samples)
- [Installation](#installation)
- [Documentation](#documentation)
- [Technical Preview Features](#technical-preview-features)
- [oneDAL and Intel&reg; DAAL](#onedal-and-intel-daal)


## Build yours high-performance data science application with oneDAL <!-- omit in toc -->

oneDAL uses all capabilities of Intel&reg; hardware, which allows you to get an significant performance boost on the classic machine learning algorithms.

We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

The current version of oneDAL provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

The size of the data is growing exponentially, as is the need for high-performance and scalable frameworks to analyze all this data and extract some benefits from it.
Besides superior performance on a single node, the oneDAL distributed computation mode also provides excellent strong and weak scaling (check charts below).

oneDAL K-means fit, strong scaling result | oneDAL K-means fit, weak scaling results
:-------------------------:|:-------------------------:
![](docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |   ![](docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

>*technical details: FPType: float32; HW: Intel Xeon Processor E5-2698 v3 @2.3GHz, 2 sockets, 16 cores per socket; SW: Intel® DAAL (2019.3), MPI4Py (3.0.0), Intel® Distribution Of Python (IDP) 3.6.8; Details available in the article https://arxiv.org/abs/1909.11822*

Refer to our [examples](#examples) and [documentation](#documentation) for more information about our API.

## Python API

oneDAL has a python API that is provided as a standalone python library called [daal4py](https://github.com/IntelPython/daal4py).
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

| *Speedups of oneDAL powered Scikit-learn over the original Scikit-learn, 28 cores, 1 thread/core* |
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
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)


## Technical Preview Features 

<<<<<<< HEAD
Technical preview features are introduced to gain early feedback from developers. A preview feature is subject to change in the future releases. Using a preview feature in a production code base is therefore strongly discouraged.
The preview features list:
- `MultiNodeBatch` for K-Means, a stepless distributed algorithm based on oneCCL
- Graph Analytics: 
	- Undirected graph without edge and vertex weights (undirected_adjacency_array_graph) - 32bit vertex index only
	- Jaccard Similarity Coefficients for all vertex pairs, a batch algorithm which processes the graph by blocks
=======
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

For more details browse [daal4py documentation](https://intelpython.github.io/daal4py/).

## oneDAL Apache Spark MLlib samples
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)

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

<<<<<<< HEAD
=======
### Installation from Source <!-- omit in toc -->
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)
See [Installation from Sources](INSTALL.md) for details.

## Examples <!-- omit in toc -->

Examples for different programming languages:

<<<<<<< HEAD
- [C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/cpp)
- [C++ SYCL*](https://github.com/oneapi-src/oneDAL/tree/master/examples/cpp_sycl)
- [Java](https://github.com/oneapi-src/oneDAL/tree/master/examples/java)

Data Examples for different computation modes:
=======
## Documentation
- [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)
- [Get Started Guide](http://oneapi-src.github.io/oneDAL/getstarted.html)
- [Developer Guide](http://oneapi-src.github.io/oneDAL/)
- [daal4py documentation](https://intelpython.github.io/daal4py/)
- [Specification](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/oneapi-dal-release-notes.html)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

## Samples <!-- omit in toc -->
Samples is an examples of how oneDAL can be used in different applications.
- [Apache Arrow](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/arrow)
- [KDB](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/kdb)
- [MPI](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mpi)
- [MySQL](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mysql)
- [Hadoop](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/hadoop)
- [Java Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/spark)
- [Scala Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/scala/spark)
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)

- [Batch](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/batch)
- [Distributed](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/distributed)
- [Online](https://github.com/oneapi-src/oneDAL/tree/master/examples/data/online)

<<<<<<< HEAD
## Documentation
=======
Technical preview features are introduced to gain early feedback from developers. A technical preview feature is subject to change in the future releases. Using a technical preview feature in a production code base is therefore strongly discouraged.

In C++ APIs, technical preview features are located in `daal::preview` and `oneapi::dal::preview` namespaces. In Java APIs, technical preview features are located in packages that have the `com.intel.daal.preview` name prefix.

The preview features list:
- Graph Analytics: 
	- Undirected graph without edge and vertex weights (`undirected_adjacency_array_graph`), where vertex indices can only be of type int32
	- Jaccard Similarity Coefficients for all pairs of vertices, a batch algorithm that processes the graph by blocks
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)

- [Get Started](http://oneapi-src.github.io/oneDAL/getstarted.html)
- [System Requirements](https://software.intel.com/en-us/articles/system-requirements-for-oneapi-data-analytics-library#)
- [oneDAL documentation](http://oneapi-src.github.io/oneDAL/)
- [Specifications](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/en-us/articles/oneapi-dal-release-notes)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

## API

Intel(R) DAAL [provides](https://software.intel.com/en-us/articles/daal-api-reference) downloadable API References for C++, Python, and Java.

<<<<<<< HEAD
You can also use [daal4py](https://github.com/IntelPython/daal4py), a simplified Python API to Intel(R) DAAL that allows fast usage of the framework suited for Data Scientists or Machine Learning users.
=======
|Product|Latest release|Branch|Resources|
|-------|--------------|------|:-------------:|
|oneDAL       |2021.1-beta10|[master](https://github.com/oneapi-src/oneDAL)</br>[rls/onedal-beta10-rls](https://github.com/oneapi-src/oneDAL/tree/rls/onedal-beta10-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Documentation](http://oneapi-src.github.io/oneDAL/)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)|
|Intel&reg; DAAL|2020 Update 3|[rls/daal-2020-u3-rls](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-u3-rls)|&nbsp;&nbsp;&nbsp;[Home page](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/daal-programming-guide/top.html)&nbsp;&nbsp;&nbsp;</br>&nbsp;&nbsp;&nbsp;[System Requirements](https://software.intel.com/content/www/us/en/develop/articles/intel-data-analytics-acceleration-library-2020-system-requirements.html)|

## Contribute <!-- omit in toc -->

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License <!-- omit in toc -->

Distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more
information.

## Security <!-- omit in toc -->

To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).
>>>>>>> 6a6439801... [DOC] Updated readme and contributing files (#1284)
