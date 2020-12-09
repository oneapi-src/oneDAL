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

.. _alg_svm:

=======================================
Support Vector Machine Classifier (SVM)
=======================================

Support Vector Machine (SVM) :capterm:`classification` is among popular classification
algorithms. It belongs to a family of generalized linear classification problems.


.. |t_math| replace::    `Training <svm_t_math_>`_
.. |t_thunder| replace:: `Thunder <svm_t_math_thunder_>`_
.. |t_smo| replace::     `SMO <svm_t_math_smo_>`_
.. |t_input| replace::   `train_input <svm_t_api_input_>`_
.. |t_result| replace::  `train_result <svm_t_api_result_>`_
.. |t_op| replace::      `train(...) <svm_t_api_>`_

.. |i_math| replace::    `Inference <svm_i_math_>`_
.. |i_smo| replace::     `SMO <svm_i_math_smo_>`_
.. |i_thunder| replace:: `Thunder <svm_i_math_thunder_>`_
.. |i_input| replace::   `infer_input <svm_i_api_input_>`_
.. |i_result| replace::  `infer_result <svm_i_api_result_>`_
.. |i_op| replace::      `infer(...) <svm_i_api_>`_

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_smo|     |t_thunder|    |t_op|   |t_input|   |t_result|
   |i_math|        |i_smo|     |i_thunder|    |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============

------------------------
Mathematical formulation
------------------------

.. _svm_t_math:

Training
--------

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, 
x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p` and a vector
of class labels :math:`y = (y_1, \ldots, y_n)`, where :math:`y_i \in \{-1, 1\}` 
describes the class to which the feature vector :math:`x_i` belongs, the problem is to build a 
two-class Support Vector Machine (SVM) classifier.

The SVM model is trained using the Sequential minimal optimization (SMO) method [Boser92]_} 
for reduced to the solution of the quadratic optimization problem

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
corresponding decision function :math:`D(x)= \sum_{k} {y_k \alpha_k K(x_k, x)} + b`,
where only those :math:`x_k` that correspond to non-zero :math:`\alpha_k` appear in the sum,
and :math:`b` is a bias. Each non-zero :math:`\alpha_k` is called a classification
coefficient and the corresponding :math:`x_k` is called a support vector.

.. _svm_t_math_smo:

Training method: *smo*
~~~~~~~~~~~~~~~~~~~~~~
In *smo* training method, all vectors from the training dataset are used for each iteration.

.. _svm_t_math_thunder:

Training method: *thunder*
~~~~~~~~~~~~~~~~~~~~~~~~~~
In *thunder* training method, the algorithm iteratively solves the convex optimization problem
with the linear constraints by selecting the fixed set of active constrains (working set) and 
applying Sequential Minimal Optimization (SMO) solver to the selected subproblem.
The description of this method is given in Algorithm [Wen2018]_. 

.. _svm_i_math:
.. _svm_i_math_smo:
.. _svm_i_math_thunder:

Inference methods: *smo* and *thunder*
--------------------------------------
*smo* and *thunder* inference methods perform prediction in the same way:

Given the SVM classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`, 
the problem is to calculate the signed value of the
decision function :math:`D(x_i)`, :math:`i=1, \ldots, r`. The sign of the
value defines the class of the feature vector, and the absolute
value of the function is a multiple of the distance between the
feature vector and the separating hyperplane.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::svm`` namespace and are available via inclusion of the
``oneapi/dal/algo/svm.hpp`` header file.
                   
Descriptor
----------
.. onedal_class:: oneapi::dal::svm::detail::v1::descriptor_base
.. onedal_class:: oneapi::dal::svm::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::svm::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::svm::task::v1

Model
-----
.. onedal_class:: oneapi::dal::svm::v1::model

.. _svm_t_api:

Training :expr:`train(...)`
--------------------------------
.. _svm_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::svm::v1::train_input


.. _svm_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::svm::v1::train_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              svm::train_result train(const Descriptor& desc, \
                                      const svm::train_input& input)

   :tparam Descriptor: SVM algorithm descriptor :expr:`svm::desc`.

   Preconditions
      | :expr:`input.data.is_empty == false`
      | :expr:`input.labels.is_empty == false`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.data.row_count == input.labels.row_count`

.. _svm_i_api:

Inference :expr:`infer(...)`
----------------------------
.. _svm_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::svm::v1::infer_input


.. _svm_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::svm::v1::infer_result

Operation
~~~~~~~~~
.. onedal_func:: oneapi::dal::v1::infer
.. function:: template <typename Descriptor> \
              svm::infer_result infer(const Descriptor& desc, \
                                      const svm::infer_input& input)

   :tparam Descriptor: SVM algorithm descriptor :expr:`svm::desc`.

   Preconditions
      | :expr:`input.data.is_empty == false`