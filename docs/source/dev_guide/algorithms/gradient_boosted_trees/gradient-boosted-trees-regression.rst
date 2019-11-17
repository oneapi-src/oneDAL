.. ******************************************************************************
.. * Copyright 2014-2019 Intel Corporation
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

Regression Gradient Boosted Trees
=================================

Gradient boosted trees regression is the special case of gradient
boosted trees. For more details, see :ref:`gb_trees`.

Details
*******

Given n feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of :math`n`
:math:`p`-dimensional feature vectors and a vector of dependent variables :math:`y = (y_1, \ldots, y_n)`, the problem is to build a gradient
boosted trees regression model that minimizes the loss function based
on the predicted and true value.

Training Stage
--------------

Gradient boosted trees regression follows the algorithmic
framework of gradient boosted trees training with following loss

.. math::
	L(y, f)=\frac{1}{2}({y-f(x))}^{2}

Prediction Stage
----------------

Given the gradient boosted trees regression model and vectors :math:`(x_1, \ldots, x_r)`, the problem is to calculate responses
for those vectors. To solve the problem for each given feature
vector :math:`x_i`, the algorithm finds the leaf node in a tree in
the ensemble, and the leaf node gives the tree response. The
algorithm result is a sum of responses of all the trees.

Batch Processing
****************

Gradient boosted trees regression follows the general workflow
described in :ref:`gb_trees` and `Regression Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction>`_.

Training
--------

In addition to parameters of the gradient boosted trees described in :ref:`gb_trees_batch`,
the gradient boosted trees regression training algorithm has the
following parameters:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - The computation method used by the gradient boosted trees regression. The only training method supported so far is the default dense method.
   * - loss
     - squared
     - Loss function type.

Prediction
----------

In addition to the common regression parameters, the gradient boosted
trees regression has the following parameters at the prediction stage:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - The computation method used by the gradient boosted trees regression. The only training method supported so far is the default dense method.
   * - numIterations
     - 0
     - An integer parameter that indicates how many trained iterations of the
       model should be used in prediction. The default value 0 denotes no
       limit. All the trained trees should be used.

Examples
********

C++: :cpp_example:`gradient_boosted_trees/gbt_reg_dense_batch.cpp`

Java*: :java_example:`gbt/GbtRegDenseBatch.java`

.. Python*: gbt_reg_dense_batch.py
