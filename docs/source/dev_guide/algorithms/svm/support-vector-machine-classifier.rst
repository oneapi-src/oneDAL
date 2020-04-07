.. ******************************************************************************
.. * Copyright 2014-2020 Intel Corporation
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

.. _svm:

Support Vector Machine Classifier
=================================

.. toctree::
   :glob:
   :maxdepth: 4


Support Vector Machine (SVM) is among popular classification
algorithms. It belongs to a family of generalized linear
classification problems. Because SVM covers binary classification
problems only in the multi-class case, SVM must be used in
conjunction with multi-class classifier methods.
SVM is a binary classifier. For a multi-class case, use
Multi-Class Classifier framework of the library.

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of size :math:`p` and a vector
of class labels :math:`y = (y_1, \ldots, y_n)`, where :math:`y_i \in \{-1, 1\}` describes the class to which the feature vector :math:`x_i`
belongs, the problem is to build a two-class Support Vector Machine
(SVM) classifier.

Training Stage
--------------
The SVM model is trained using the Boser method [Boser92]_
reduced to the solution of the quadratic optimization problem

.. math::
      \underset{\alpha }{\mathrm{min}}\frac{1}{2}{\alpha }^{T}Q\alpha -{e}^{T}\alpha

with :math:`0 \leq \alpha_i \leq C`, :math:`i = 1, \ldots, n`, :math:`y^T \alpha = 0`,

where :math:`e` is the vector of ones, :math:`C` is the upper bound of the
coordinates of the vector :math:`\alpha`, :math:`Q` is a symmetric matrix of size :math:`n \times n`
with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`, and :math:`K(x,y)` is a kernel function.

Working subset of α updated on each iteration of the algorithm is
based on the Working Set Selection (WSS) 3 scheme [Fan05]_.
The scheme can be optimized using one of these techniques or both:

      -  **Cache**:      
         the implementation can allocate a predefined amount of memory
         to store intermediate results of the kernel computation.

      -  **Shrinking**:
         the implementation can try to decrease the amount of kernel
         related computations (see [Joachims99]_).

The solution of the problem defines the separating hyperplane and
corresponding decision function :math:`D(x)= \sum_{k} {y_k \alpha_k K(x_k, x)} + b` where only those :math:`x_k` that
correspond to non-zero :math:`\alpha_k` appear in the sum, and :math:`b` is a
bias. Each non-zero :math:`\alpha_k` is called a classification
coefficient and the corresponding :math:`x_k` is called a support
vector.

Prediction Stage
----------------

Given the SVM classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`, the problem is to calculate the signed value of the
decision function :math:`D(x_i)`, :math:`i=1, \ldots, r`. The sign of the
value defines the class of the feature vector, and the absolute
value of the function is a multiple of the distance between the
feature vector and separating hyperplane.

Usage of Training Alternative
*****************************

To build a Support Vector Machine (SVM) Classifier model using methods of the Model Builder class of SVM Classifier,
complete the following steps:

- Create an SVM Classifier model builder using a constructor with the required number of support vectors and features.
- In any sequence:

  - Use the ``setSupportVectors``, ``setClassificationCoefficients``, and ``setSupportIndices`` methods to add
    pre-calculated support vectors, classification coefficients, and support indices (optional), respectively, to the model.
    For each method specify random access iterators to the first and the last element
    of the corresponding set of values [ISO/IEC 14882:2011 § 24.2.7].
  - Use ``setBias`` to add a bias term to the model.

- Use the ``getModel`` method to get the trained SVM Classifier model.
- Use the ``getStatus`` method to check the status of the model building process. 
  If ``DAAL_NOTHROW_EXCEPTIONS`` macros is defined, the status report contains the list of errors
  that describe the problems API encountered (in case of API runtime failure).

.. note::

   If after calling the getModel method you use the ``setBias``, ``setSupportVectors``, ``setClassificationCoefficients``, or ``setSupportIndices``
   methods, coefficients, the initial model will be automatically updated with the new set of parameters.

Examples
--------

.. tabs::

  .. tab:: C++
  
   :cpp_example:`svm_two_class_model_builder.cpp <svm/svm_two_class_model_builder.cpp>`

  .. tab:: Java*

   :java_example:`SVMTwoClassModelBuilder.java <svm/SVMTwoClassModelBuilder.java>`
  
  .. tab:: Python*

   :daal4py_example:`svm_two_class_model_builder.py`

Batch Processing
****************

SVM classifier follows the general workflow described in
:ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to Usage Model:
Training and Prediction.

At the training stage, SVM classifier has the following parameters:

+-----------------------+-----------------------+-----------------------+
| Parameter             | Default Value         | Description           |
+=======================+=======================+=======================+
| algorithmFPType       | float                 | The floating-point    |
|                       |                       | type that the         |
|                       |                       | algorithm uses for    |
|                       |                       | intermediate          |
|                       |                       | computations. Can be  |
|                       |                       | float or double.      |
+-----------------------+-----------------------+-----------------------+
| method                | defaultDense          | The computation       |
|                       |                       | method used by the    |
|                       |                       | SVM classifier. The   |
|                       |                       | only training method  |
|                       |                       | supported so far is   |
|                       |                       | the Boser method.     |
+-----------------------+-----------------------+-----------------------+
| nClasses              | 2                     | The number of         |
|                       |                       | classes.              |
+-----------------------+-----------------------+-----------------------+
| C                     | 1                     | Upper bound in        |
|                       |                       | conditions of the     |
|                       |                       | quadratic             |
|                       |                       | optimization problem. |
+-----------------------+-----------------------+-----------------------+
| accuracyThreshold     | 0.001                 | The training          |
|                       |                       | accuracy.             |
+-----------------------+-----------------------+-----------------------+
| tau                   | 1.0e-6                | Tau parameter of the  |
|                       |                       | WSS scheme.           |
+-----------------------+-----------------------+-----------------------+
| maxIterations         | 1000000               | Maximal number of     |
|                       |                       | iterations for the    |
|                       |                       | algorithm.            |
+-----------------------+-----------------------+-----------------------+
| cacheSize             | 8000000               | Size of cache in      |
|                       |                       | bytes for storing     |
|                       |                       | values of the kernel  |
|                       |                       | matrix. A non-zero    |
|                       |                       | value enables use of  |
|                       |                       | a cache optimization  |
|                       |                       | technique.            |
+-----------------------+-----------------------+-----------------------+
| doShrinking           | true                  | A flag that enables   |
|                       |                       | use of a shrinking    |
|                       |                       | optimization          |
|                       |                       | technique.            |
+-----------------------+-----------------------+-----------------------+
| kernel                | Pointer to an object  | The kernel function.  |
|                       | of the KernelIface    | By default, the       |
|                       | class                 | algorithm uses a      |
|                       |                       | linear kernel.        |
+-----------------------+-----------------------+-----------------------+

Prediction
----------

For a description of the input and output, refer to Usage Model:
Training and Prediction.

At the prediction stage, SVM classifier has the following parameters:


+-----------------------+-----------------------+-----------------------+
| Parameter             | Default Value         | Description           |
+=======================+=======================+=======================+
| algorithmFPType       | float                 | The floating-point    |
|                       |                       | type that the         |
|                       |                       | algorithm uses for    |
|                       |                       | intermediate          |
|                       |                       | computations. Can be  |
|                       |                       | float or double.      |
+-----------------------+-----------------------+-----------------------+
| method                | defaultDense          | Performance-oriented  |
|                       |                       | computation method,   |
|                       |                       | the only prediction   |
|                       |                       | method supported by   |
|                       |                       | the algorithm.        |
+-----------------------+-----------------------+-----------------------+
| nClasses              | 2                     | The number of         |
|                       |                       | classes.              |
+-----------------------+-----------------------+-----------------------+
| kernel                | Pointer to object of  | The kernel function.  |
|                       | the KernelIface class | By default, the       |
|                       |                       | algorithm uses a      |
|                       |                       | linear kernel.        |
+-----------------------+-----------------------+-----------------------+

Examples
--------

.. tabs::

  .. tab:: C++

   Batch Processing:

   -  :cpp_example:`svm_two_class_dense_batch.cpp <svm/svm_two_class_dense_batch.cpp>`
   -  :cpp_example:`svm_two_class_csr_batch.cpp <svm/svm_two_class_csr_batch.cpp>`

  .. tab:: Java*
   
   Batch Processing:

   -  :java_example:`SVMTwoClassDenseBatch.java <svm/SVMTwoClassDenseBatch.java>`
   -  :java_example:`SVMTwoClassCSRBatch.java <svm/SVMTwoClassCSRBatch.java>`

.. Python*:

.. -  svm_two_class_dense_batch.py
.. -  svm_two_class_csr_batch.py
.. -  svm_two_class_metrics_dense_batch.py

Performance Considerations
**************************

For the best performance of the SVM classifier, use homogeneous
numeric tables if your input data set is homogeneous or SOA numeric
tables otherwise.

Performance of the SVM algorithm greatly depends on the cache size
cacheSize. Larger cache size typically results in greater
performance. For the best SVM algorithm performance, use cacheSize
equal to :math:`n^2 \cdot \text{sizeof(algorithmFPType)}`. However, avoid
setting the cache size to a larger value than the number of bytes
required to store :math:`n^2` data elements because the algorithm
does not fully utilize the cache in this case.

.. include:: ../../../opt-notice.rst



