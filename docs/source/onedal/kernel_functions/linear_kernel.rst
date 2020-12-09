.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. default-domain:: cpp

.. _alg_linear_kernel:

=============
Linear kernel
=============

The linear kernel is the simplest kernel function for pattern analysis.

.. |c_math| replace::   `dense <linear_kernel_c_math_>`_
.. |c_dense| replace::  `dense <linear_kernel_c_math_dense_>`_
.. |c_input| replace::  `compute_input <linear_kernel_c_api_input_>`_
.. |c_result| replace:: `compute_result <linear_kernel_c_api_result_>`_
.. |c_op| replace::     `compute(...) <linear_kernel_c_api_>`_

=============== ============= ======== =========== ==================
 **Operation**  **Computational methods**   **Programming Interface**
--------------- --------------------------- -------------------------
   |c_math|      |c_dense|     |c_op|   |c_input|          |c_result|
=============== ============= ======== =========== ==================

------------------------
Mathematical formulation
------------------------

.. _linear_kernel_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), 
\ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, x_{mp})`, 
the problem is to compute the linear kernel function :math:`K(x_i,, y_i)` for
any pair of input vectors: :math:`K(x_i, y_i) = k {X_i}^T y_i + b`.

.. _linear_kernel_c_math_dense:

---------------------
Programming Interface
---------------------
All types and functions in this section shall be declared in the
``oneapi::dal::linear_kernel`` namespace and be available via inclusion of the
``oneapi/dal/algo/linear_kernel.hpp`` header file.
                   
Descriptor
----------
.. onedal_class:: oneapi::dal::linear_kernel::detail::v1::descriptor_base
.. onedal_class:: oneapi::dal::linear_kernel::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_kernel::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_kernel::task::v1

.. _linear_kernel_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _linear_kernel_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::linear_kernel::v1::compute_input


.. _linear_kernel_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::linear_kernel::v1::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              linear_kernel::compute_result compute(const Descriptor& desc, \
                                      const linear_kernel::compute_input& input)

   :tparam Descriptor: Linear Kernel algorithm descriptor :expr:`linear_kernel::desc`.

   Preconditions
      | :expr:`input.data.is_empty == false`
