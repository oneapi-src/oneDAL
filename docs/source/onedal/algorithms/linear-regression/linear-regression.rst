.. ******************************************************************************
.. * Copyright 2024 Intel Corporation
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

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_linear-regression:

=================
Linear Regression
=================

.. include::  ../../../includes/linear-regression/linear-regression-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _linreg_t_math:

Training
---------

Given a set :math:`(x_{11}, \ldots, x_{1p}, y_1, \ldots, x_{n1}, \ldots, x_{np}, y_n)` of
training data, :math:`n \gg p`. The matrix :math:`X` of size :math:`n \times p` contains
observations :math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` of independent
variables.

To estimate the coefficients :math:`(\beta_{0j}, \ldots, \beta_{pj})`
Normal Equation system method can be used.

.. _linreg_p_math:

Partial Training
----------------

Given a block :math:`(x_{11}, \ldots, x_{1p}, y_1, \ldots, x_{n1}, \ldots, x_{np}, y_n)` of
training data, :math:`n \gg p`. The matrix :math:`X` of size :math:`n \times p` contains
observations :math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` of independent
variables.

To compute crossproduct of Linear Regression Normal Equation system method can be used.

.. _linreg_f_math:

Finalize Training
-----------------

Given a partial result with partial products.

To compute crossproduct of Linear Regression Normal Equation system method can be used.


.. _linreg_i_math:

Inference
---------
Linear regression based prediction is done for input vector
using the equation :math:`y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p`
for each :math:`j=1, \ldots, k`.

.. _norm_eq:

Computation method: *norm_eq*
-----------------------------
The method computes means, variance-covariance, or correlation matrix for the dense data.
This is the default and the only method supported.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Linear Regression <api_linear-regression>`.

-----------
Online mode
-----------

The algorithm supports online mode.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode (only on GPU).
