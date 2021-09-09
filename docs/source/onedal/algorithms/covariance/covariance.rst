.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. _alg_covariance:

================
Covariance
================

.. include::  ../../../includes/covariance/covariance-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _covariance_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` :math:`p`-dimensional feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`,
the problem is to compute the sample means or the
covariance matrix or the correlation matrix:

.. list-table::
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m(1), \ldots , m(p))`, where :math:`m\left(j\right)=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m(i))(x_{kj}-m(j))`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`

.. _covariance_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the means or the
variance-covariance matrix or the correlation matrix

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Covariance <api_covariance>`.
