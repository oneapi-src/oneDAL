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

.. include::  ../../../includes/covariance/linear-regression-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _linreg_t_math:

Training
---------

Given a dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`,
the means is a :math:`1 \times p` matrix, the covariance and the correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m_{1}, \ldots , m_{p})`, where :math:`m_{j}=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix (sample)
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Covariance matrix (maximum likelihood)
     - :math:`Cov' = (v'_{ij})`, where :math:`v'_{ij}=\frac{1}{n}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`


.. _linreg_p_math:

Partial Training
----------------

Given a dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`,
the means is a :math:`1 \times p` matrix, the covariance and the correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m_{1}, \ldots , m_{p})`, where :math:`m_{j}=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix (sample)
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Covariance matrix (maximum likelihood)
     - :math:`Cov' = (v'_{ij})`, where :math:`v'_{ij}=\frac{1}{n}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`

.. _linreg_f_math:

Finalize Training
-----------------

Given a dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`,
the means is a :math:`1 \times p` matrix, the covariance and the correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m_{1}, \ldots , m_{p})`, where :math:`m_{j}=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix (sample)
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Covariance matrix (maximum likelihood)
     - :math:`Cov' = (v'_{ij})`, where :math:`v'_{ij}=\frac{1}{n}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`


.. _linreg_i_math:

Inference
---------
Given the inference data set :math:`X^{'} = \{ x^{'}_1, \ldots, x^{'}_m \}` with :math:`m` feature vectors of dimension :math:`p`,
and the :math:`r \times p` transformation matrix :math:`T` produced at the training stage,
the problem is to transform :math:`X^{'}` to the :math:`m \times r` matrix :math:`X^{''} = \{ x^{''}_1, \ldots, x^{''}_m \}`, where :math:`x^{''}_{i}` is an
:math:`r`-dimensional transformed observation.

Each individual observation :math:`x^{'}_{i}` can be transformed by applying the following linear
transformation [Lang87]_ defined by the matrix :math:`T`,

.. math::
   :label: x_transform

   x^{''}_{i} = x^{'}_{i} T^{*}, \quad 1 \leq i \leq m, \quad T^{*} = transpose(T).



.. _linreg_t_math_norm_eq:

Computation method: *dense*
---------------------------
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
