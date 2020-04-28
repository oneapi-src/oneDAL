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

Pivoted QR Decomposition
========================

Given the matrix :math:`X` of size :math:`n \times p`, the problem is to compute the QR decomposition
with column pivoting :math:`XP = QR`, where

- :math:`Q` is an orthogonal matrix of size :math:`n \times n`
- :math:`R` is a rectangular upper triangular matrix of size :math:`n \times p`
- :math:`P` is a permutation matrix of size :math:`n \times n`

The library requires :math:`n > p`. In this case:

.. math::

    XP = QR = [Q_1, Q_2] \cdot \begin{bmatrix} R_1 \\ 0 \end{bmatrix} = Q_1 R_1

where the matrix :math:`Q_1` has the size :math:`n \times p` and :math:`R_1` has the size :math:`p \times p`.

Batch Processing
****************

At this moment, the description of batch processing for Pivoted QR Decomposition is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-11>`_.
