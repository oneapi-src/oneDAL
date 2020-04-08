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

Kernel Functions
----------------

.. toctree::
   :glob:
   :maxdepth: 4

Kernel functions form a class of algorithms for pattern analysis. The
main characteristic of kernel functions is a distinct approach to
this problem. Instead of reducing the dimension of the original data,
kernel functions map the data into higher-dimensional spaces in order
to make the data more easily separable there.


Linear Kernel
=============

A linear kernel is the simplest kernel function.

Problem Statement
*****************

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, x_{mp})`, the problem is to
compute the linear kernel function :math:`K(x_i,, y_i)` for
any pair of input vectors: :math:`K(x_i, y_i) = k {X_i}^T y_i + b`.


Batch Processing
****************

Algorithm Input
+++++++++++++++

The linear kernel function accepts the input described below. Pass
the Input ID as a parameter to the methods that provide input for
your algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - X
     - Pointer to the :math:`n \times p` numeric table that represents the matrix X. This table can be an object of any class derived from NumericTable.
   * - Y
     - Pointer to the :math:`m \times p` numeric table that represents the matrix Y. This table can be an object of any class derived from NumericTable.

Algorithm Parameters
++++++++++++++++++++

The linear kernel function has the following parameters:

.. list-table::
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

       + defaultDense - default performance-oriented method
       + fastCSR - performance-oriented method for CSR numeric tables

   * - ComputationMode
     - matrixMatrix
     - Computation mode for the kernel function. Can be:

       + vectorVector - compute the kernel function for given feature vectors :math:`x_i` and :math:`y_j`
       + matrixVector - compute the kernel function for all vectors in the set :math:`X` and a given feature vector :math:`y_j`
       + matrixMatrix - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`. In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

   * - rowIndexX
     - :math:`0`
     - Index i of the vector in the set :math:`X` for the vectorVector computation mode.
   * - rowIndexY
     - :math:`0`
     - Index :math:`j` of the vector in the set :math:`Y` for the vectorVector or matrixVector computation mode.
   * - rowIndexResult
     - :math:`0`
     - Row index in the values numeric table to locate the result of the computation for the vectorVector computation mode.
   * - :math:`k`
     - :math:`1`
     - The coefficient :math:`k` of the linear kernel.
   * - :math:`b`
     - :math:`0`
     - The coefficient :math:`b` of the linear kernel.

Algorithm Output
++++++++++++++++

The linear kernel function calculates the results described below.
Pass the Result ID as a parameter to the methods that access the
results of your algorithm.


.. list-table::
   :header-rows: 1
   :align: left


   * - Result ID
     - Result
   * - values
     - Pointer to the :math:`n \times m` numeric table with the values of the kernel
       function. By default, this result is an object of the
       HomogenNumericTable class, but you can define the result as an object of
       any class derived from NumericTable except PackedSymmetricMatrix,
       PackedTriangularMatrix, and CSRNumericTable.

Examples
++++++++

.. tabs::

  .. tab:: C++

    - :cpp_example:`kernel_func_lin_dense_batch.cpp <kernel_function/kernel_func_lin_dense_batch.cpp>`
    - :cpp_example:`kernel_func_lin_csr_batch.cpp <kernel_function/kernel_func_lin_csr_batch.cpp>`

  .. tab:: Java*

    - :java_example:`KernelFuncLinDenseBatch.java <kernel_function/KernelFuncLinDenseBatch.java>`
    - :java_example:`KernelFuncLinCSRBatch.java <kernel_function/KernelFuncLinCSRBatch.java>`

.. Python*:

.. - kernel_func_lin_dense_batch.py
.. - kernel_func_lin_csr_batch.py


Radial Basis Function Kernel
============================

The Radial Basis Function (RBF) kernel is a popular kernel function
used in kernelized learning algorithms.

Problem Statement
*****************

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, x_{mp})`,
the problem is to compute the RBF kernel function :math:`K(x_i,, y_i)` for any pair of input vectors:

.. math::
   K\left({x}_{i},{y}_{j}\right)=exp\left(-\frac{{\left(‖{x}_{i}-{y}_{j}‖\right)}^{2}}{2{\sigma }^{2}}\right)

Batch Processing
****************

Algorithm Input
+++++++++++++++

The RBF kernel accepts the input described below. Pass the Input
ID as a parameter to the methods that provide input for your
algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - :math:`X`
     - Pointer to the :math:`n \times p` numeric table that represents the matrix :math:`X`. This table can be an object of any class derived from NumericTable.
   * - :math:`Y`
     - Pointer to the :math:`m \times p` numeric table that represents the matrix :math:`Y`. This table can be an object of any class derived from NumericTable.

Algorithm Parameters
++++++++++++++++++++

The RBF kernel has the following parameters:

.. list-table::
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

       + defaultDense - default performance-oriented method
       + fastCSR - performance-oriented method for CSR numeric tables

   * - ComputationMode
     - matrixMatrix
     - Computation mode for the kernel function. Can be:

       + vectorVector - compute the kernel function for given feature vectors :math:`x_i` and :math:`y_j`
       + matrixVector - compute the kernel function for all vectors in the set :math:`X` and a given feature vector :math:`y_j`
       + matrixMatrix - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`. In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

   * - rowIndexX
     - :math:`0`
     - Index :math:`i` of the vector in the set :math:`X` for the vectorVector computation mode.
   * - rowIndexY
     - :math:`0`
     - Index :math:`j` of the vector in the set :math:`Y` for the vectorVector or matrixVector computation mode.
   * - rowIndexResult
     - :math:`0`
     - Row index in the values numeric table to locate the result of the computation for the vectorVector computation mode.
   * - sigma
     - :math:`0`
     - The coefficient :math:`\sigma` of the RBF kernel.

Algorithm Output
++++++++++++++++

The RBF kernel calculates the results described below. Pass the
Result ID as a parameter to the methods that access the results of
your algorithm.

.. list-table::
   :header-rows: 1
   :align: left


   * - Result ID
     - Result
   * - values
     - Pointer to the :math:`n \times m` numeric table with the values of the kernel
       function. By default, this result is an object of the
       HomogenNumericTable class, but you can define the result as an object of
       any class derived from NumericTable except PackedSymmetricMatrix,
       PackedTriangularMatrix, and CSRNumericTable.

Examples
********

.. tabs::

  .. tab:: C++

    Batch Processing:

    - :cpp_example:`kernel_func_rbf_dense_batch.cpp <kernel_function/kernel_func_rbf_dense_batch.cpp>`
    - :cpp_example:`kernel_func_rbf_csr_batch.cpp <kernel_function/kernel_func_rbf_csr_batch.cpp>`

  .. tab:: Java*

    Batch Processing:

    - :java_example:`KernelFuncRbfDenseBatch.java <kernel_function/KernelFuncRbfDenseBatch.java>`
    - :java_example:`KernelFuncRbfCSRBatch.java <kernel_function/KernelFuncRbfCSRBatch.java>`

.. Python*:

.. - kernel_func_rbf_dense_batch.py
.. - kernel_func_rbf_csr_batch.py
