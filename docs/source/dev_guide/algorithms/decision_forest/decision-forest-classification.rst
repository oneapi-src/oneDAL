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

.. _df_classification:

Classification Decision Forest
------------------------------

.. toctree::
   :glob:
   :maxdepth: 4

Decision forest classifier is a special case of the :ref:`decision_forest` model.

Details
=======

Given :math:`n` feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` 
of :math:`n` :math:`p`-dimensional feature vectors, a vector of class labels :math:`y = (y_1, \ldots, y_n)`, 
where :math:`y_i \in \{0, 1, \ldots, C-1\}` describes the class to which the feature vector :math:`x_i` belongs
and :math:`C` is the number of classes, the problem is to build a decision
forest classifier.

Training Stage
**************

Decision forest classifier follows the algorithmic framework of
decision forest training with Gini impurity metrics as impurity
metrics, that are calculated as follows:

.. math::
	{I}_{Gini}\left(D\right)=1-\sum _{i=0}^{C-1}{p}_{i}^{2}

where :math:`p_i` is the fraction of observations in the subset :math:`D` that belong to the :math:`i`-th class.

Prediction Stage
****************

Given decision forest classifier and vectors :math:`x_1, \ldots, x_r`, 
the problem is to calculate the labels for those
vectors. To solve the problem for each given query vector :math:`x_i`, 
the algorithm finds the leaf node in a tree in the
forest that gives the classification response by that tree. The
forest chooses the label y taking the majority of trees in the
forest voting for that label.

Out-of-bag Error
****************

Decision forest classifier follows the algorithmic framework for
calculating the decision forest out-of-bag (OOB) error, where
aggregation of the out-of-bag predictions in all trees and
calculation of the OOB error of the decision forest is done as
follows:

-

 For each vector :math:`x_i` in the dataset :math:`X`, predict its label :math:`\hat{y_i}` by having the majority of votes from the trees that
 contain :math:`x_i` in their OOB set, and vote for that label.

-

 Calculate the OOB error of the decision forest :math:`T` as the average
 of misclassifications:

 .. math::
	OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{y_i \in {D}^{\text{'}}}I\{y_i \ne \hat{y_i}\}\text{,where }{D}^{\text{'}}={\bigcup }_{b=1}^{B}\overline{D_b}.

-  If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`: :math:`OOB(x_i) = I\{{y}_{i}\ne \hat{{y}_{i}}\}`


Variable Importance
*******************

The library computes *Mean Decrease Impurity* (MDI) importance
measure, also known as the *Gini importance* or *Mean Decrease
Gini*, by using the Gini index as impurity metrics.


Batch Processing
================

Decision forest classification follows the general workflow described in :ref:`decision_forest` and `Classification Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_.

Training
********

In addition to the parameters of a classifier (see `Classification Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_) and decision forest parameters
described in :ref:`df_batch`, the training algorithm for decision forest classification has the
following parameters:


.. list-table::
   :widths: 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - ?
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - The computation method used by the decision forest classification. The
       only prediction method supported so far is the default dense method.
   * - nClasses
     - Not applicable.
     - The number of classes. A required parameter.

Output
******

Decision forest classification calculates the result of regression
and decision forest. For more details, refer to :ref:`df_batch` and `Classification Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_.

Prediction
**********

For the description of the input and output, refer to `Classification Usage Model <https://software.intel.com/en-us/daal-programming-guide-usage-model-training-and-prediction-1>`_.

In addition to the parameters of a classifier, decision forest
classification has the following parameters at the prediction stage:

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
     - The computation method used by the decision forest classification. The
       only prediction method supported so far is the default dense method.
   * - nClasses
     - Not applicable.
     - The number of classes. A required parameter.

Examples
********

C++: :cpp_example:`decision_forest/df_cls_dense_batch.cpp`

Java*: :java_example:`decision_forest/DfClsDenseBatch.java`

.. Python*: df_cls_dense_batch.py

