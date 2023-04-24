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

For a dataset :math:`X_{n \times p}` with :math:`n` observations and :math:`p` features,
the covariance and the correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::

    :widths: 10 60
    :header-rows: 1
    :align: left

    * - Statistic
      - Definition
    * - Means
      - :math:`M = (M_j)`,:math:`j = \overline{1,p}`, :math:`M_j = \frac{1}{n}\sum _{i} X_{ij}`
    * - Covariance matrix
      - :math:`S = (S_{ij})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`, :math:`S_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(X_{ki} - M_i)(X_{kj}-M_j)`
    * - Correlation matrix
      - :math:`C = C_{ij}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`,:math:`C_{ij} = \frac{S_{ij}}{\sqrt{S_{ii}\cdot S_{jj}}}`

.. _covariance_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the means or the variance-covariance matrix or the correlation matrix for dense data.
This is also the default and the only method supported.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Covariance <api_covariance>`.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SMPD mode (only on GPU).
