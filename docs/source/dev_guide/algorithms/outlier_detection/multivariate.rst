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

Multivariate Outlier Detection
==============================

In multivariate outlier detection methods, the observation point is the entire feature vector.

Details
*******

Given a set :math:`X` of :math:`n` feature vectors
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p`,
the problem is to identify the vectors that do not belong to the underlying distribution
(see [Ben2005]_ for exact definitions of an outlier).

The multivariate outlier detection method takes into account dependencies between features.
This method can be parametric, assumes a known underlying distribution for the data set, and defines an outlier region
such that if an observation belongs to the region, it is marked as an outlier.
Definition of the outlier region is connected to the assumed underlying data distribution.

The following is an example of an outlier region for multivariate outlier detection:

.. math::

    \text{Outlier}(\alpha_n, M_n, \Sigma_n) = \{x: \sqrt{(x - M_n) \sum _{n}{-1} (x - M_n)} > g(n, \alpha_n) \}

where :math:`M_n` and \Sigma_n are (robust) estimates of the vector of means and variance-covariance matrix computed for a given data set,
:math:`\alpha_n` is the confidence coefficient, and :math:`g(n, \alpha_n)` defines the limit of the region.

Batch Processing
****************

At this moment, the description of batch processing for Multivariate Outlier Detection is only available in
`Developer Guide for Intel(R) DAAL
<https://software.intel.com/content/www/us/en/develop/documentation/daal-programming-guide/top/algorithms/analysis/multivariate-outlier-detection/batch-processing-15.html>`_.
