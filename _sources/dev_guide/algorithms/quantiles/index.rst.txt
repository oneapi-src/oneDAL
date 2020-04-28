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

Quantile
========

Quantile is an algorithm to analyze the distribution of observations.
Quantiles are the values that divide the distribution so that a given portion of observations is below the quantile.

Details
*******

Given a set :math:`X` of :math:`p` features :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
and the quantile orders :math:`\beta = \beta_1, \ldots, \beta_m`,
the problem is to compute :math:`z_{ik}` that meets the following conditions:

.. math::
    P\{ \xi_i \leq z_{ik} \} \geq \beta_k

.. math::
    P\{\xi_i > z_{ik} \} \leq 1 - \beta_k
    
In the equations above:

- :math:`x_i = (x_{1i}, \ldots, x_{ni})` are observations of a random variable :math:`\xi_i` that represents the :math:`i`-th feature
- :math:`P` is the probability measure
- :math:`i = 1, \ldots, p`
- :math:`k = 1, \ldots, m`

Batch Processing
****************

At this moment, the description of batch processing is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-1>`_.
