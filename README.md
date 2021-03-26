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

# Intel&reg; oneAPI Data Analytics Library <!-- omit in toc -->

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](#documentation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](#support)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Samples](#samples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/oneapi-src.oneDAL?branchName=master)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=5&branchName=master) [![License](https://img.shields.io/github/license/oneapi-src/oneDAL.svg)](https://github.com/oneapi-src/oneDAL/blob/master/LICENSE) [![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/oneapi-src/oneDAL/discussions)

Intel&reg; oneAPI Data Analytics Library (oneDAL) is a powerful machine learning library that helps speed up big data analysis. oneDAL solvers are also used in [Intel Distribution for Python](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) for scikit-learn optimization.

Intel&reg; oneAPI Data Analytics Library is an extension of Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL).


## Table of Contents <!-- omit in toc -->
- [Python API](#python-api)
  - [Scikit-learn patching](#scikit-learn-patching)
  - [Distributed multi-node mode](#distributed-multi-node-mode)
- [oneDAL Apache Spark MLlib samples](#onedal-apache-spark-mllib-samples)
- [Installation](#installation)
- [Documentation](#documentation)
- [Support](#support)
- [Technical Preview Features](#technical-preview-features)
- [oneDAL and Intel&reg; DAAL](#onedal-and-intel-daal)


## Build your high-performance data science application with oneDAL <!-- omit in toc -->

oneDAL uses all capabilities of Intel&reg; hardware, which allows you to get a significant performance boost for the classic machine learning algorithms.

We provide highly optimized algorithmic building blocks for all stages of data analytics: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

oneDAL also provides Data Parallel C++ (DPC++) API extensions to the traditional C++ interfaces.

The size of the data is growing exponentially as does the need for high-performance and scalable frameworks to analyze all this data and benefit from it.
Besides superior performance on a single node, oneDAL also provides distributed computation mode that shows excellent results for strong and weak scaling:

oneDAL K-Means fit, strong scaling result | oneDAL K-Means fit, weak scaling results
:-------------------------:|:-------------------------:
![](docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |   ![](docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

>*Technical details: FPType: float32; HW: Intel Xeon Processor E5-2698 v3 @2.3GHz, 2 sockets, 16 cores per socket; SW: Intel® DAAL (2019.3), MPI4Py (3.0.0), Intel® Distribution Of Python (IDP) 3.6.8; Details available in the article https://arxiv.org/abs/1909.11822*

Refer to our [examples](#examples) and [documentation](#documentation) for more information about our API.

## Python API

oneDAL has a Python API that is provided as a standalone Python library called [daal4py](https://github.com/IntelPython/daal4py).

The example below shows how daal4py can be used to calculate K-Means clusters:

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

With a Python API provided by daal4py, you can create scikit-learn compatible estimators, transformers, or clusterers that are powered by oneDAL and are nearly as efficient as native programs.

| *Speedup of oneDAL-powered scikit-learn over the original scikit-learn, 28 cores, 1 thread/core* |
|:--:|
| ![](docs/readme-charts/IDP%20scikit-learn%20accelearation%20compared%20with%20stock%20scikit-learn.png) |
| *Technical details: FPType: float32; HW: Intel(R) Xeon(R) Platinum 8276L CPU @ 2.20GHz, 2 sockets, 28 cores per socket; SW: scikit-learn 0.22.2, Intel® DAAL (2019.5), Intel® Distribution Of Python (IDP) 3.7.4; Details available in the article https://medium.com/intel-analytics-software/accelerate-your-scikit-learn-applications-a06cacf44912* |

daal4py have an API that matches scikit-learn API.
This framework allows you to speed up your existing projects by changing one line of code.

```python
from daal4py.sklearn.svm import SVC
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

svm = SVC(kernel='rbf', gamma='scale', C = 0.5).fit(X, y)
print(svm.score(X, y))
```

In addition, daal4py provides an option to replace some scikit-learn methods by oneDAL solvers, which makes it possible to get a performance gain **without any code changes**. This approach is the basis of Intel distribution for Python scikit-learn. You can patch the stock scikit-learn by using the following command-line flag:
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

Data scientists often require different tools for analysis of regular and big data. daal4py offers various processing models, which makes it easy to enable distributed multi-node mode.

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

<img align="right" style="display:inline;" height=300 width=550 src="docs/readme-charts/intel%20oneDAL%20Spark%20samples%20vs%20Apache%20Spark%20MLlib.png"></a>

oneDAL provides Scala and Java interfaces that match Apache Spark MlLib API and use oneDAL solvers under the hood. This implementation allows you to get a 3-18X increase in performance compared to the default Apache Spark MLlib.

>*Technical details: FPType: double; HW: 7 x m5.2xlarge AWS instances; SW: Intel DAAL 2020 Gold, Apache Spark 2.4.4, emr-5.27.0; Spark config num executors 12, executor cores 8, executor memory 19GB, task cpus 8*

Check the [samples](#samples) tab for more details.

## Installation

You can install oneDAL:

- from [oneDAL home page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) as a part of Intel&reg; oneAPI Base Toolkit.
- from [GitHub\*](https://github.com/oneapi-src/oneDAL/releases).

### Installation from Source <!-- omit in toc -->
See [Installation from Sources](INSTALL.md) for details.

## Examples <!-- omit in toc -->

Beside C++ and Python API, oneDAL also provides APIs for DPC++ and Java:
- [C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp)
- [oneAPI C++](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/cpp)
- [oneAPI DPC++](https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/dpc)
- [Java](https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/java)
- [Python](https://github.com/IntelPython/daal4py/tree/master/examples)

## Documentation
- [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/system-requirements-for-oneapi-data-analytics-library.html)
- [Get Started Guide](http://oneapi-src.github.io/oneDAL/onedal/get-started.html#onedal-get-started)
- [Developer Guide and Reference](http://oneapi-src.github.io/oneDAL/)
- [daal4py documentation](https://intelpython.github.io/daal4py/)
- [Specification](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)
- [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/oneapi-dal-release-notes.html)
- [Known Issues](https://oneapi-src.github.io/oneDAL/notes/known_issues.html)

Refer to GitHub Wiki to browse [the full list of oneDAL and daal4py resources](https://github.com/oneapi-src/oneDAL/wiki/Resources).

## Support

Ask questions and engage in discussions with oneDAL developers, contributers, and other users through the following channels:

- [GitHub Discussions](https://github.com/oneapi-src/oneDAL/discussions)
- [Community Forum](https://community.intel.com/t5/Intel-oneAPI-Data-Analytics/bd-p/oneapi-data-analytics-library)

You may reach out to project maintainers privately at onedal.maintainers@intel.com.

### Security <!-- omit in toc -->

To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

### Contribute <!-- omit in toc -->

Report issues and make feature requests using [GitHub Issues](https://github.com/oneapi-src/oneDAL/issues).

We welcome community contributions, so check our [contributing guidelines](CONTRIBUTING.md) to learn more.

### Feedback <!-- omit in toc -->

Use [GitHub Wiki](https://github.com/oneapi-src/oneDAL/wiki/Feedback) to provide feedback about oneDAL.

## Samples <!-- omit in toc -->
Samples are examples of how oneDAL can be used in different applications:
- [Apache Arrow](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/arrow)
- [KDB](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/kdb)
- [MPI](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mpi)
- [MySQL](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/mysql)
- [Hadoop](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/hadoop)
- [Java Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/java/spark)
- [Scala Spark](https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/scala/spark)

## Technical Preview Features

Technical preview features are introduced to gain early feedback from developers. A technical preview feature is subject to change in the future releases. Using a technical preview feature in a production code base is therefore strongly discouraged.

In C++ APIs, technical preview features are located in `daal::preview` and `oneapi::dal::preview` namespaces. In Java APIs, technical preview features are located in packages that have the `com.intel.daal.preview` name prefix.

The preview features list:
- Graph Analytics:
	- Undirected graph without edge and vertex weights (`undirected_adjacency_vector_graph`), where vertex indices can only be of type int32
	- Jaccard Similarity Coefficients for all pairs of vertices, a batch algorithm that processes the graph by blocks
  - Local and Global Triangle Counting

## oneDAL and Intel&reg; DAAL

Intel&reg; oneAPI Data Analytics Library is an extension of Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL).

This repository contains branches corresponding to both oneAPI and classical versions of the library. We encourage you to use oneDAL located under the `master` branch.

|Product|Latest release|Branch|
|-------|--------------|------|
|oneDAL       |2021.2|[master](https://github.com/oneapi-src/oneDAL)</br>[rls/2021.2-rls](https://github.com/oneapi-src/oneDAL/tree/rls/2021.2-rls)|
|Intel&reg; DAAL|2020 Update 3|[rls/daal-2020-u3-rls](https://github.com/oneapi-src/oneDAL/tree/rls/daal-2020-u3-rls)|


## License <!-- omit in toc -->

oneDAL is distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more
information.

[oneMKL FPK microlibs](https://github.com/oneapi-src/oneDAL/releases/tag/Dependencies)
are distributed under Intel Simplified Software License.
Refer to [third-party-programs-mkl.txt](third-party-programs-mkl.txt) for details.
