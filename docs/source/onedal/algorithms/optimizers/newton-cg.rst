.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
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

.. _alg_newton_cg:

===================
Newton-CG optimizer
===================

.. include::  ../../../includes/optimizers/newton-cg-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _newton_cg_c_math:

Computing
---------

Newton-CG optimizer also known as the hessian-free optimizer is used to minimize convex function 
without calculating the hessian matrix but using a hessian product matrix operator. The descent direction
in the Newton method is calculated using the formula :math:`d_k = -H_k^{-1} g_k` where :math:`g_k, H_k` are
the gradient and hessian matrix of the loss function on the :math:`k`-th iteration. In Newton-CG method
Conjugate Gradients solver is used to find the approximate solution to the equation :math:`H_k d_k = -g_k`.
Conjugate Gradients solver can find solution to the system of linear equations :math:`Ax = b` taking 
vector :math:`b` and functor :math:`f(p) = Ap` as input.  


.. _newton_cg_c_math_dense:

Computation method: *dense*
---------------------------
The method defines the Newton-CG optimizer, which is used in other algorithms
for convex optimization. There are no separate computation modes to minimize
function manually.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Newton-CG optimizer <api_newton_cg>`.
