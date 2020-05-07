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

Cosine Distance Matrix
======================

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
of dimension Lmath:`p`, 
the problem is to compute the symmetric :math:`n \times n` matrix :math:`D_{\text{cos}} = (d_{ij})`
of distances between feature vectors, where

.. math::

    d_{ij} = 1 - \frac
    {\sum_{k=1}^{p} x_{ik} x_{jk}}
    {\sqrt{ \sum_{k=1}^{p} x_{ik}^2 } 
    \sqrt{ \sum_{k=1}^{p} x_{jk}^2 }}

.. math::
    i = \overline{1, n}

.. math::
    j = \overline{1, n}


Batch Processing
****************

At this moment, the description of batch processing for Cosine Distance Matrix is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-3>`_.

Performance Considerations
**************************

To get the best overall performance when computing the cosine distance matrix:

- If input data is homogeneous, provide the input data and store results in homogeneous numeric tables
  of the same type as specified in the ``algorithmFPType`` class template parameter.
- If input data is non-homogeneous, use :ref:`AOS <aos_layout>` layout rather than SOA layout.

.. include:: ../../../opt-notice.rst
