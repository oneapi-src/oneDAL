.. ******************************************************************************
.. * Copyright contributors to the oneDAL project
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp

CPU Features Dispatching
^^^^^^^^^^^^^^^^^^^^^^^^

For each algorithm oneDAL provides several code paths for x86-64-compatibe instruction
set architectures.

Following architectures are currently supported:
- Streaming SIMD Extensions 2 (SSE2)
- Streaming SIMD Extensions 4.2 (SSE4.2)
- Advanced Vector Extensions 2 (AVX2)
- Advanced Vector Extensions 512 (AVX-512)

The particular code path is chosen at runtime based on the underlying hardware characteristics.

This chapter describes how the code is organized to support this variety of instruction sets.

Algorithm Implementation Options
********************************

Besides the instruction sets architecture, an algorithm in oneDAL might have various implementation
options. The description of those options is provided below for better understanding of the oneDAL
code structure and conventions.

Computational Tasks
-------------------

An algorithm might have various tasks to compute. The most common options are:

- `Classification https://oneapi-src.github.io/oneDAL/onedal/glossary.html#term-Classification`_,
- `Regression https://oneapi-src.github.io/oneDAL/onedal/glossary.html#term-Regression`.

Computational Stages
--------------------

An algorithm might have ``training`` and ``inference`` computaion stages aimed
to train a model on the input dataset and compute the inference results respectively.

Computational Methods
---------------------

An algorithm can support several methods for the same type of computations.
For example, kNN algorithm supports
`brute_force <https://oneapi-src.github.io/oneDAL/onedal/algorithms/nearest-neighbors/knn.html#knn-t-math-brute-force>`_
and `kd_tree <https://oneapi-src.github.io/oneDAL/onedal/algorithms/nearest-neighbors/knn.html#knn-t-math-kd-tree>`_
methods for algorithm training and inference.

Computational Modes
-------------------

oneDAL can provide several computaional modes for an algorithm.
See `Computaional Modes <https://oneapi-src.github.io/oneDAL/onedal/programming-model/computational-modes.html>`_
chapter for details.

Folders and Files
*****************

Consider you are working on some algorithm ``Abc`` in oneDAL.

The part of the implementation of this algorithms that is running on CPU should be located in
`cpp/daal/src/algorithms/abc` folder.

Consider it provides:

- ``classification`` and ``regression`` learning tasks;
- ``training`` and ``inference`` stages;
- ``method1`` and ``method2`` for the ``training`` stage and only ``method1`` for ``inference`` stage;
- only batch computational mode.

Then the `cpp/daal/src/algorithms/abc` folder should contain at least the following files:

| cpp/daal/src/algorithms/abc
| |-- abc_classification_predict_method1_batch_fpt_cpu.cpp
| |-- abc_classification_predict_impl.i
| |-- abc_classification_predict_kernel.h
| |-- abc_classification_train_method1_batch_fpt_cpu.cpp
| |-- abc_classification_train_method2_batch_fpt_cpu.cpp
| |-- abc_classification_train_impl.i
| |-- abc_classification_train_kernel.h
| |-- abc_regression_predict_method1_batch_fpt_cpu.cpp
| |-- abc_regression_predict_impl.i
| |-- abc_regression_predict_kernel.h
| |-- abc_regression_train_method1_batch_fpt_cpu.cpp
| |-- abc_regression_train_method2_batch_fpt_cpu.cpp
| |-- abc_regression_train_impl.i
| |-- abc_regression_train_kernel.h

Alternative variant of the folder structure to avoid storing too much files within a single folder
can be:

| cpp/daal/src/algorithms/abc
| |-- classification
| |    |-- abc_classification_predict_method1_batch_fpt_cpu.cpp
| |    |-- abc_classification_predict_impl.i
| |    |-- abc_classification_predict_kernel.h
| |    |-- abc_classification_train_method1_batch_fpt_cpu.cpp
| |    |-- abc_classification_train_method2_batch_fpt_cpu.cpp
| |    |-- abc_classification_train_impl.i
| |    |-- abc_classification_train_kernel.h
| |-- regression
| |    |-- abc_regression_predict_method1_batch_fpt_cpu.cpp
| |    |-- abc_regression_predict_impl.i
| |    |-- abc_regression_predict_kernel.h
| |    |-- abc_regression_train_method1_batch_fpt_cpu.cpp
| |    |-- abc_regression_train_method2_batch_fpt_cpu.cpp
| |    |-- abc_regression_train_impl.i
| |    |-- abc_regression_train_kernel.h

The names of the files stay the same in this case, just the folders layout differs.
