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

.. _df_regression:

Regression Decision Forest
--------------------------

.. toctree::
   :glob:
   :maxdepth: 4

Decision forest regression is a special case of the :ref:`decision_forest` model.

Details
=======

Given n feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of :math:`n`
:math:`p`-dimensional feature vectors and vector of dependent variables :math:`y = (y_1, \ldots, y_n)`,
the problem is to build a decision
forest regression model that minimizes the Mean-Square Error (MSE)
between the predicted and true value.

Training Stage
**************

Decision forest classifier follows the algorithmic framework of
decision forest training with variance as impurity metrics,
calculated as follows:

:math:`{I}_{var}\left(D\right)= \frac{1}{N-1} \sum _{i=0}^{N-1}{\left({y}_{i}-\overline{y}\right)}^{2}`, where :math:`N=|D|` and :math:`\overline{y}=\text{ }\frac{1}{N}\text{ }\sum _{i=0}^{N-1}{y}_{i}`.


Prediction Stage
****************

Given decision forest regression model and vectors :math:`x_1, \ldots, x_r`, the problem is to calculate the responses for those
vectors. To solve the problem for each given query vector :math:`x_i`, the algorithm finds the leaf node in a tree in the
forest that gives the response by that tree as the mean of
dependent variables. The forest predicts the response as the mean
of responses from trees.


Out-of-bag Error
****************

Decision forest regression follows the algorithmic framework for
calculating the decision forest out-of-bag (OOB) error, where
aggregation of the out-of-bag predictions in all trees and
calculation of the OOB error of the decision forest is done as
follows:

-  For each vector :math:`x_i` in the dataset :math:`X`, predict its response :math:`\hat{y_i}`
   as the mean of prediction from the trees that contain :math:`x_i` in their OOB set:

   :math:`\hat{y_i} = \frac{1}{{|B}_{i}|}\sum _{b=1}^{|B_i|}\hat{y_{ib}}`, where :math:`B_i= \bigcup{T_b}: x_i \in \overline{D_b}` and :math:`\hat{y_{ib}}` is the result of prediction
   :math:`x_i` by :math:`T_b`.

-  Calculate the OOB error of the decision forest T as the Mean-Square Error (MSE):

   .. math::
		OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{{y}_{i} \in {D}^{\text{'}}}\sum {(y_i-\hat{y_i})}^{2}, \text{where } {D}^{\text{'}}={\bigcup}_{b=1}^{B}\overline{{D}_{b}}

-  If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`:

   .. math::
		OOB(x_i) = {(y_i-\hat{y_i})}^{2}


Batch Processing
================

Decision forest regression follows the general workflow described in :ref:`decision_forest`.

Training
********

For the description of the input and output, refer to :ref:`regression_usage_model`.

In addition to the decision forest parameters described in :ref:`df_batch`, 
the training algorithm for decision forest regression has the following parameters:

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
     - The computation method used by the decision forest regression. The
       only prediction method supported so far is the default dense method.

Output
******

In addition to the output of regression described in  :ref:`regression_usage_model`,
decision forest regression calculates the result of decision forest.
For more details, refer to :ref:`df_batch`.

Prediction
**********

For the description of the input and output, refer to :ref:`regression_usage_model`.

In addition to the parameters of regression, decision forest
regression has the following parameters at the prediction stage:

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
     - The computation method used by the decision forest regression. The
       only prediction method supported so far is the default dense method.

Examples
********

.. tabs::

  .. tab:: C++

    Batch Processing:

    - :cpp_example:`df_reg_dense_batch.cpp <decision_forest/df_reg_dense_batch.cpp>`

  .. tab:: Java*

    Batch Processing:

    - :java_example:`DfRegDenseBatch.java <decision_forest/DfRegDenseBatch.java>`

.. Python*: df_reg_dense_batch.py

