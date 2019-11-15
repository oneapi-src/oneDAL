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

Classification Gradient Boosted Trees
=====================================

Gradient boosted trees classification is the special case of gradient
boosted trees. For more details, see Gradient Boosted Trees.

Details
*******

Given n feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of n
p-dimensional feature vectors and a vector of class labels :math:`y = (y_1, \ldots, y_n)`, where
:math:`y_i \in \{0, 1, \ldots, C-1\}` and
C is the number of classes, which describes the class to which the
feature vector :math:`x_i` belongs, the problem is to build a gradient
boosted trees classifier.

Training Stage
--------------

Gradient boosted trees classification follows the algorithmic
framework of gradient boosted trees training. For a classification
problem with K classes, K regression trees are constructed on each
iteration, one for each output class. The loss function is
cross-entropy (multinomial deviance):

.. math::
	L(y,f) = -\sum_{k=1}^{K}(I(y=k)\ln{p_{k}(x)})

where :math:`{p}_{k}(x)=\mathrm{ }\frac{{e}^{{f}_{k}\left(x\right)}}{\sum _{i=1}^{K}{e}^{{f}_{i}\left(x\right)}}`

Binary classification is a special case when single regression tree is trained on each iteration. The loss function is

.. math::
	L(y,f) = -(y\cdot \ln{\sigma(f)} + (1-y)\ln(1- \sigma(f)))

where :math:`\sigma(f)=\frac{1}{1+ {e}^{-f}}`

Prediction Stage
----------------

Given the gradient boosted trees classifier model and vectors :math:`(x_1, \ldots, x_r)`, the problem is to calculate labels for
those vectors. To solve the problem for each given feature vector
:math:`x_i`, the algorithm finds the leaf node in a tree in the
ensemble, and the leaf node gives the tree response. The algorithm
computes a sum of responses of all the trees for each class and
chooses the label y corresponding to the class with the maximal
response value (highest class probability).


Batch Processing
****************

Gradient boosted trees classification follows the general workflow
described in :ref:`gb_trees` and `Classification Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_

Training
--------

In addition to parameters of the gradient boosted trees described
in :ref:`gb_trees_batch`, the gradient boosted trees classification training algorithm has
the following parameters:

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
     - The computation method used by the gradient boosted trees regression.
       The only training method supported so far is the default dense method.
   * - nClasses
     - Not applicable.
     - The number of classes. A required parameter.
   * - loss
     - crossEntropy
     - Loss function type.

Prediction
----------

In addition to the parameters of a classifier, the gradient boosted
trees classifier has the following parameters at the prediction stage:

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
     - The computation method used by the gradient boosted trees regression.
       The only training method supported so far is the default dense method.
   * - nClasses
     - Not applicable.
     - The number of classes. A required parameter.
   * - numIterations
     - 0
     - An integer parameter that indicates how many trained iterations of the
       model should be used in prediction. The default value 0 denotes no
       limit. All the trained trees should be used.

Examples
********

C++: :cpp_example:`gradient_boosted_trees/gbt_cls_dense_batch.cpp`

Java*: :java_example:`gbt/GbtClsDenseBatch.java`

.. Python*: gbt_cls_dense_batch.py
