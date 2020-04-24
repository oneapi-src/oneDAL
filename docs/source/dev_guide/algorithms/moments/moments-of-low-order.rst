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

Moments of Low Order
====================

Moments are basic quantitative measures of data set characteristics such as location and dispersion.
|short_name| computes the following low order characteristics: 

- minimums/maximums
- sums
- means
- sums of squares
- sums of squared differences from the means
- second order raw moments
- variances
- standard deviations
- variations

Details
*******

Given a set :math:`X` of :math:`n` feature vectors 
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p`, 
the problem is to compute the following sample characteristics for each feature in the data set:

.. list-table::
   :widths: 20 60
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Minimum
     - :math:`min(j) = \smash{\displaystyle \min_i } \{x_{ij}\}`
   * - Maximum
     - :math:`max(j) = \smash{\displaystyle \max_i } \{x_{ij}\}`
   * - Sum
     - :math:`s(j) = \sum_i x_{ij}`
   * - Sum of squares
     - :math:`s_2(j) = \sum_i x_{ij}^2`
   * - Means
     - :math:`m(j) = \frac {s(j)} {n}`
   * - Second order raw moment
     - :math:`a_2(j) = \frac {s_2(j)} {n}`
   * - Sum of squared difference from the means
     - :math:`\text{SDM}(j) = \sum_i (x_{ij} - m(j))^2`
   * - Variance
     - :math:`k_2(j) = \frac {\text{SDM}(j) } {n - 1}`
   * - Standard deviation
     - :math:`\text{stdev}(j) = \sqrt {k_2(j)}`
   * - Variation coefficient
     - :math:`V(j) = \frac {\text{stdev}(j)} {m(j)}`

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1
   
   computation-batch.rst
   computation-online.rst
   computation-distributed.rst

DPC++ Examples
**************

Batch Processing:

- :ref:`low_order_moms_dense_batch.cpp`

Online Processing:

- :ref:`low_order_moms_dense_online.cpp`
