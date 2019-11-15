.. ******************************************************************************
.. * Copyright 2014-2019 Intel Corporation
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

Linear and Ridge Regressions Computation
****************************************

Batch Processing
================

Linear and ridge regressions in the batch processing mode follow the
general workflow described in Usage Model: Training and
Prediction.

Training
--------

For a description of the input and output, refer to Usage Model:
Training and Prediction.

The following table lists parameters of linear and ridge
regressions at the training stage. Some of these parameters or
their values are specific to a linear or ridge regression
algorithm.

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1
   :align: left

   * - Parameter
     - Algorithm
     - Default Value
     - Description
   * - algorithmFPType
     - any
     - float
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be float or double.
   * - method
     - - linear regression
       - ridge regression
     - defaultDense
     - - Available methods for linear regression training:
         -  defaultDense - the normal equations method
         -  qrDense - the method based on QR decomposition
       - Default computation method used by the ridge regression. The only method
         supported at the training stage is the normal equations method.
   * - ridgeParameters
     - ridge regression
     - Numeric table of size :math:`1 \times 1` that contains the default ridge parameter
       equal to :math:`1`.
     - The numeric table of size :math:`1 \times k` (:math:`k` is the number of dependent variables)
       or :math:`1 \times 1`. The contents of the table depend on its size:

       -  :math:`size = 1 \times k`: values of the ridge parameters :math:`\lambda_j` for :math:`j = 1, \ldots, k`.

       -  :math:`size = 1 \times 1`: the value of the ridge parameter for each dependent
          variable :math:`\lambda_1 = \ldots = \lambda_k`.

         This parameter can be an object of any class derived from NumericTable,
         except for PackedTriangularMatrix, PackedSymmetricMatrix, and
         CSRNumericTable.
   * - interceptFlag
     - any
     - true
     - A flag that indicates a need to compute :math:`\beta_{0j}`.


Prediction
----------

For a description of the input and output, refer to `Usage Model: Training and Prediction <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_.

At the prediction stage, linear and ridge regressions have the
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
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be float or double.
   * - method
     - defaultDense
     - Default performance-oriented computation method, the only method
       supported by the regression based prediction.

Examples
========

C++:

-  :cpp_example:`linear_regression/lin_reg_norm_eq_dense_batch.cpp`
-  :cpp_example:`linear_regression/lin_reg_qr_dense_batch.cpp`
-  :cpp_example:`ridge_regression/ridge_reg_norm_eq_dense_batch.cpp`

Java*:

-  :java_example:`linear_regression/LinRegNormEqDenseBatch.java`
-  :java_example:`linear_regression/LinRegQRDenseBatch.java`
-  :java_example:`ridge_regression/RidgeRegNormEqDenseBatch.java`

.. Python*:

.. -  linear_regression_norm_eq_dense_batch.py
.. -  lin_reg_qr_dense_batch.py
.. -  ridge_reg_norm_eq_dense_batch.py
.. -  lin_reg_metrics_dense_batch.py
