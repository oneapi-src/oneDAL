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

================
Kernel Functions
================

A kernel function takes input vectors from original (:math:`p`-dimensional) space
and returns dot product of the vectors in :math:`s`-dimensional feature space.
More formally, if we have :math:`x,y \in \mathbb{R}^p`, and
:math:`\phi \in \mathbb{R}^p \leftarrow \mathbb{R}^s`, then kernel function is

.. math::
   K(x, y) = <\phi (x), \phi (y)>
.. _kernel_func_def:

In case, when :math:`\phi (x) = x`, the kernel is linear.
Kernels are used in SVM model, but for some tasks it could be used separately to transform vectors
from one space to another.

The following table describes current device support:

+--------------+------+------+
| Kernel type  | CPU  | GPU  |
+==============+======+======+
| Linear       | Yes  | Yes  |
+--------------+------+------+
| Polynomial   | Yes  | No   |
+--------------+------+------+
| RBF          | Yes  | Yes  |
+--------------+------+------+
| Sigmoid      | Yes  | No   |
+--------------+------+------+


.. toctree::
   :titlesonly:

   linear-kernel.rst
   polynomial-kernel.rst
   rbf-kernel.rst
   sigmoid-kernel.rst

.. rubric:: Examples: Linear Kernel

.. include::  ../../../includes/kernel-functions/linear-kernel-examples.rst

.. rubric:: Examples: Polynomial Kernel

.. include::  ../../../includes/kernel-functions/polynomial-kernel-examples.rst

.. rubric:: Examples: RBF Kernel

.. include:: ../../../includes/kernel-functions/rbf-kernel-examples.rst

.. rubric:: Examples: Sigmoid Kernel

.. include:: ../../../includes/kernel-functions/sigmoid-kernel-examples.rst
