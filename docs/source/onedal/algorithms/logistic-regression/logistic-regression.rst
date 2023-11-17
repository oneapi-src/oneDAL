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

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_logreg:

===================
Logistic Regression
===================

.. include:: ../../../includes/logistic-regression/logistic-regression-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _logreg_t_math:

Training
--------

:math:``

Given :math:`n` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,
x_n=(x_{n1},\ldots,x_{np})\}` of size :math:`p` 
and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\} \in {0,1}` 
the problem is to fit the model weights :math:`w=\{w_0, \ldots, w_p\}` to minimize
Logistic Loss :math:`L(X, w, y) = \sum_{i = 1}^{n} -y_i \log(prob_i) - (1 - y_i) \log(prob_i)`, 
where :math:`prob_i = \sigma(w_0 + \sum_{j=1}^{p} w_j x_{i, j})` - predicted probabilities, 
:math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` - sigmoid function. Note that probabilities are binded to interval :math:`[\epsilon, 1 - \epsilon]` to avoid problems with 
computing log function (:math:`\epsilon=10^{-7}` if float type is used and :math:`10^{-15}` otherwise)

.. _logreg_t_math_dense_batch:

As Logistic Loss is a convex function one of the convex iterative solvers can be used for minimization.

.. tabs::
    .. group-tab:: Newton-CG
        Newton-CG solver description

.. _logreg_i_math:

Given :math:`r` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,
x_r=(x_{r1},\ldots,x_{rp})\}` of size :math:`p` the problem is to calculate probabilities of 
objects corresponding to these feature vectors belonging to each of the classes and calculate 
the most probable class labels.

.. _logreg_i_math_dense_batch:

The probabilities are calculated using this formula 
:math:`prob_i = \sigma(w_0 + \sum_{j=1}^{p} w_j x_{i, j})`, where 
:math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` - is a sigmoid function. 
If the probability is bigger then :math:`0.5` then class label is set to :math:`1`,
otherwise to :math:`0`.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Logistic Regression <api_logreg>`.
