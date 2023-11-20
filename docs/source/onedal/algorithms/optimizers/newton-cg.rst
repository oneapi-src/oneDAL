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


.. _newton_cg_c_math_dense:

Computation method: *dense*
---------------------------
The method defines Newton-CG optimizer, which is used in other algorithms
for the convex optimization. There are no separate computation mode to minimize
function manually.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Newton-CG optimizer <api_newton_cg>`.
