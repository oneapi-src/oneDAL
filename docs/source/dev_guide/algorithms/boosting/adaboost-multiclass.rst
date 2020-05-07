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

AdaBoost Multiclass Classifier
==============================

AdaBoost (short for "Adaptive Boosting") is a popular boosting classification algorithm.
The AdaBoost algorithm performs well on a variety of data sets except some noisy data ([Friedman98]_, [Zhu2005]_).
The library supports two methods for the algorithms:

- SAMME, or Stagewise Additive Modeling using a Multi-class Exponential loss function [Zhu2005]_
- SAMME.R, which is a modification of SAMME method for Real-valued returned probabilities from weak learner

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
of size :math:`p`, a vector of class labels :math:`y = (y_1, \ldots, y_n)` 
where :math:`y_i \in K = \{-1, 1\}` in case of binary classification and
:math:`y_i \in K = \{ 0, \ldots, C-1 \}`, where :math:`C` is a number of classes,
describes the class :math:`t` the feature vector :math:`x_i` belongs to, 
and :math:`h_t` is a weak learner algorithm, the problem is to build an AdaBoost classifier.

SAMME method
************

At this moment, the description of the SAMME method is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-samme-method>`_.

SAMME.R method
**************

At this moment, the description of the SAMME.R method is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-samme-r-method>`_.

Batch Processing
****************

At this moment, the description of batch processing for AdaBoost MultiClass is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-122>`_.
