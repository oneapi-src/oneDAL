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

Cholesky Decomposition
======================

Cholesky decomposition is a matrix factorization technique that decomposes a symmetric positive-definite matrix
into a product of a lower triangular matrix and its conjugate transpose.

Because of numerical stability and superior efficiency in comparison with other methods,
Cholesky decomposition is widely used in numerical methods for solving symmetric linear systems.
It is also used in non-linear optimization problems, Monte Carlo simulation, and Kalman filtration.

Details
*******

Given a symmetric positive-definite matrix :math:`X` of size :math:`p \times p`, the problem is to compute the Cholesky decomposition
:math:`X = {LL}^T`, where :math:`L` is a lower triangular matrix.

Batch Processing
****************

At this moment, the description of batch processing for Cholesky decomposition is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-9>`_.

Performance Considerations
**************************

To get the best overall performance when Cholesky decomposition:

- If input data is homogeneous, for input matrix :math:`X` and output matrix :math:`L` use homogeneous numeric tables
  of the same type as specified in the ``algorithmFPType`` class template parameter.
- If input data is non-homogeneous, use :ref:`AOS <aos_layout>` layout rather than SOA layout.

.. include:: ../../../opt-notice.rst
