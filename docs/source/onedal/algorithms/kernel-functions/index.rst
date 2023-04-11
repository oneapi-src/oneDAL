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

A kernel function is function that takes input vectors from original space
and returns dot product of the vectors in the feature space.
More formally, if we have :math:`x,y \in X`, and :math:`\phi \in X \leftarrow \mathbb{R}^n`,
then kernel function

.. math::
   K(x, y) = <\phi (x), \phi (y)>
.. _kernel_func_def:

In case, when :math:`\phi (x) = x`, the kernel is linear.
Kernels are used in SVM model, but for some tasks it could be used separately to transform vectors
from one space to another.

The following tables describes current device support:
.. list-table:: Device support
   :widths: 50 10 10
   :header-rows: 1

   * - Kernel name
     - CPU
     - GPU
   * - Linear
     - Yes
     - Yes
   * - Polynomial
     - Yes
     - No
   * - RBF
     - Yes
     - Yes
   * - Sigmoid
     - Yes
     - Yes

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
