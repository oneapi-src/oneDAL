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

Singular Value Decomposition
============================

.. toctree::
   :glob:
   :maxdepth: 4


Singular Value Decomposition (SVD) is one of matrix factorization
techniques. It has a broad range of applications including
dimensionality reduction, solving linear inverse problems, and data
fitting.

Details
*******

Given the matrix :math:`X` of size :math:`n \times p`, the problem is to compute the
Singular Value Decomposition (SVD) :math:`X = U \Sigma V^t`, where:

   -  :math:`U` is an orthogonal matrix of size :math:`n \times n`

   -  :math:`\Sigma` is a rectangular diagonal matrix of size :math:`n \times p` with non-negative values on the diagonal, called singular values

   -  :math:`V_t` is an orthogonal matrix of size :math:`p \times p`

Columns of the matrices :math:`U` and :math:`V` are called left and right singular vectors, respectively.

Batch and Online Processing
***************************

Online processing computation mode assumes that the data arrives in blocks :math:`i = 1, 2, 3, \ldots \text{nblocks}`.

Algorithm Input
---------------

The SVD algorithm accepts the input described below. Pass the Input ID as a parameter to the methods that provide input for your algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the numeric table that represents:

       - For batch processing, the entire :math:`n \times p` matrix :math:`X` to be factorized.
       - For online processing, the :math:`n_i \times p` submatrix of :math:`X` that represents 
         the current data block in the online processing mode. Note that each current data block must have sufficient size: :math:`n_i > p`.

       The input can be an object of any class derived from NumericTable.


Algorithm Parameters
--------------------

The SVD algorithm has the following parameters:

.. list-table::
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - method
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - leftSingularMatrix
     - ``requiredInPackedForm``
     - Specifies whether the matrix of left singular vectors is required. Can be:

       - ``notRequired`` - the matrix is not required
       - ``requiredInPackedForm`` - the matrix in the packed format is required

   * - rightSingularMatrix
     - ``requiredInPackedForm``
     - Specifies whether the matrix of left singular vectors is required. Can be:

       - ``notRequired`` - the matrix is not required
       - ``requiredInPackedForm`` - the matrix in the packed format is required

Algorithm Output
----------------

The SVD algorithm calculates the results described below. Pass the Result ID as a parameter to the methods that access the results of your algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - singularValues
     - Pointer to the :math:`1 \times p` numeric table with singular values (the diagonal of the matrix :math:`\Sigma`). 
       By default, this result is an object of the HomogenNumericTable class, but you can define the result as an object of any class 
       derived from NumericTable except PackedSymmetricMatrix, PackedTriangularMatrix, and CSRNumericTable.
   * - leftSingularMatrix
     - Pointer to the :math:`n \times p` numeric table with left singular vectors (matrix :math:`U`). Pass ``NULL`` if left singular vectors are not required. 
       By default, this result is an object of the HomogenNumericTable class, but you can define the result as an object of any class 
       derived from NumericTable except PackedSymmetricMatrix, PackedTriangularMatrix, and CSRNumericTable.
   * - rightSingularMatrix
     - Pointer to the :math:`p \times p` numeric table with right singular vectors (matrix :math:`V`). Pass ``NULL`` if right singular vectors are not required.
       By default, this result is an object of the HomogenNumericTable class, but you can define the result as an object of any class 
       derived from NumericTable except PackedSymmetricMatrix, PackedTriangularMatrix, and CSRNumericTable.
   

Distributed Processing
**********************

At this moment, the description of distributed processing for Singular Value Decomposition
is only available in `Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-distributed-processing-5>`_.

Examples
********

.. tabs::

  .. tab:: C++

    Batch Processing:

    - :cpp_example:`svd_dense_batch.cpp <svd/svd_dense_batch.cpp>`

    Online Processing:

    - :cpp_example:`svd_dense_online.cpp <svd/svd_dense_online.cpp>`

  .. tab:: Java*

    Batch Processing:

    - :java_example:`SVDDenseBatch.java <svd/SVDDenseBatch.java>`

    Online Processing:
    
    - :java_example:`SVDDenseOnline.java <svd/SVDDenseOnline.java>`


Performance Considerations
**************************

To get the best overall performance of singular value decomposition
(SVD), for input, output, and auxiliary data, use homogeneous numeric
tables of the same type as specified in the algorithmFPType class
template parameter.

Online Processing
-----------------

SVD in the online processing mode is at least as computationally
complex as in the batch processing mode and has high memory
requirements for storing auxiliary data between calls to the
compute() method. On the other hand, the online version of SVD may
enable you to hide the latency of reading data from a slow data
source. To do this, implement load prefetching of the next data
block in parallel with the compute() method for the current block.

Online processing mostly benefits SVD when the matrix of left
singular vectors is not required. In this case, memory
requirements for storing auxiliary data goes down from :math:`O(p \cdot n)` to :math:`O(p \cdot p \cdot \text{nblocks})`.
