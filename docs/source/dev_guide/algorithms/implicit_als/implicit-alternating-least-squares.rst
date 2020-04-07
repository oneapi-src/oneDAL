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

Implicit Alternating Least Squares
==================================

.. toctree::
   :glob:
   :maxdepth: 4


The library provides the Implicit Alternating Least Squares (implicit
ALS) algorithm [Fleischer2008],
based on collaborative filtering.

Details
*******

Given the input dataset :math:`R=\left\{{r}_{ui}\right\}` of size :math:`m \times n`, where m is the number
of users and n is the number of items, the problem is to
train the Alternating Least Squares (ALS) model represented as two
matrices: :math:`X` of size :math:`m \times f`, and :math:`Y` of size :math:`f \times n`, where :math:`f` is the
number of factors. The matrices :math:`X` and :math:`Y` are the factors of
low-rank factorization of matrix :math:`R`:

.. math::
	R\approx X\cdot Y

Initialization Stage
--------------------

Initialization of the matrix Y can be done using the following method: for each :math:`i = 1, \ldots, n`
:math:`{y}_{1i}=\frac{1}{m}\sum _{u=1}^{m}{r}_{ui}` and :math:`y_{ki}` are independent
random numbers uniformly distributed on the interval :math:`(0,1)`, :math:`k = 2, \ldots, f`.

Training Stage
--------------

The ALS model is trained using the implicit ALS algorithm
[Hu2008]
by minimizing the following cost function:

.. math::
	\underset{{x}_{*},{y}_{*}}{\mathrm{min}}\underset{u,i}{\mathrm{\Sigma }}{c}_{ui}{\left({p}_{ui}-{x}_{u}^{T}{y}_{i}\right)}^{2}+\lambda \left(\underset{u}{\mathrm{\Sigma }}{n}_{{x}_{u}}{‖{x}_{u}‖}^{2}+\underset{i}{\mathrm{\Sigma }}{m}_{{y}_{i}}{‖{y}_{i}‖}^{2}\right),

where:

- :math:`{p}_{ui}` indicates the preference of user u of item i: :math:`p_{ui} = \lbrace \begin{array}{c}1, & {r}_{ui}>\epsilon,\\ 0, & {r}_{ui}\le \epsilon\end{array}`

- :math:`\epsilon` is the threshold used to define the preference values. :math:`\epsilon = 0` is the only threshold valu supported so far.
- :math:`{c}_{ui}=1+\alpha r_{ui}`, :math:`c_{ui}` measures the confidence in observing :math:`p_{ui}`
- :math:`\alpha` is the rate of confidence
- :math:`r_{ui}` is the element of the matrix :math:`R`
- :math:`\lambda` is the parameter of the regularization
- :math:`{n}_{{x}_{u}}`, :math:`{m}_{{y}_{i}}` denote the number of ratings of user :math:`u` and item :math:`i` respectively

Prediction Stage
----------------

.. rubric:: Prediction of Ratings

Given the trained ALS model and the matrix :math:`D` that describes for
which pairs of factors :math:`X` and :math:`Y` the rating should be computed, the
system calculates the matrix of recommended ratings Res: :math:`{res}_{ui}=\sum _{j=1}^{f}{x}_{uj}{y}_{ji}`, if :math:`{d}_{ui}\ne 0`,
:math:`u=1,\ldots,m`; :math:`i=1,\ldots n`.


Initialization
**************

Batch Processing
----------------

Input
+++++

Initialization of item factors for the implicit ALS algorithm
accepts the input described below. Pass the Input ID as a
parameter to the methods that provide input for your algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the :math:`m \times n` numeric table with the mining data. The input can
       be an object of any class derived from NumericTable except
       PackedTriangularMatrix and PackedSymmetricMatrix.

Parameters
++++++++++

Initialization of item factors for the implicit ALS algorithm has the
following parameters:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - Available computation methods:

       + defaultDense - performance-oriented method
       + fastCSR - performance-oriented method for CSR numeric tables

   * - nFactors
     - :math:`10`
     - The total number of factors.
   * - engine
     - SharePtr< engines:: mt19937:: Batch>()
     - Pointer to the random number generator engine that is used internally at the initialization step.

Output
++++++

Initialization of item factors for the implicit ALS algorithm
calculates the result described below. Pass the Result ID as a
parameter to the methods that access the results of your algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - model
     - The model with initialized item factors. The result can only be an object of the Model class.

Computation
***********

Batch Processing
----------------

Training
++++++++

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the training stage, the implicit ALS recommender has the
following parameters:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - Available computation methods:

       + defaultDense - performance-oriented method
       + fastCSR - performance-oriented method for CSR numeric tables

   * - nFactors
     - :math:`10`
     - The total number of factors.
   * - maxIterations
     - :math:`5`
     - The number of iterations.
   * - alpha
     - :math:`40`
     - The rate of confidence.
   * - lambda
     - :math:`0.01`
     - The parameter of the regularization.
   * - preferenceThreshold
     - :math:`0`
     - Threshold used to define preference values. :math:`0` is the only threshold supported so far.

Prediction
++++++++++

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the prediction stage, the implicit ALS recommender has the
following parameters:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - Performance-oriented computation method, the only method supported by the algorithm.

Examples
++++++++

.. tabs::

  .. tab:: C++

    Batch Processing:

    - :cpp_example:`impl_als_dense_batch.cpp <implicit_als/impl_als_dense_batch.cpp>`
    - :cpp_example:`impl_als_csr_batch.cpp <implicit_als/impl_als_csr_batch.cpp>`

  .. tab:: Java*

    Batch Processing:
    
    - :java_example:`ImplAlsDenseBatch.java <implicit_als/ImplAlsDenseBatch.java>`
    - :java_example:`ImplAlsCSRBatch.java <implicit_als/ImplAlsCSRBatch.java>`

.. Python*:

.. - impl_als_dense_batch.py
.. - impl_als_csr_batch.py


Performance Considerations
**************************

To get the best overall performance of the implicit ALS recommender:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is sparse, use CSR numeric tables.

.. include:: ../../../opt-notice.rst



