.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. _alg_df:

==================================================
Decision Forest Classification and Regression (DF)
==================================================

Decision Forest (DF) :capterm:`classification` and :capterm:`regression` algorithms are based on an ensemble of 
tree-structured classifiers, which are known as decision trees. Decision forest is built 
using the general technique of bagging, a bootstrap aggregation, and a random choice of features. Decision tree is
a binary tree graph. Its internal (split) nodes represent a decision function used to select the child node at the 
prediction stage. Its leaf, or terminal, nodes represent the corresponding response values, which are the result 
of the prediction from the tree. For more details, see [Breiman84]_ and [Breiman2001]_.

.. |t_math| replace:: `Training <df_t_math_>`_
.. |t_dense| replace:: `Dense <df_t_math_dense_>`_
.. |t_hist| replace:: `Hist <df_t_math_hist_>`_
.. |t_input| replace:: `train_input <df_t_api_input_>`_
.. |t_result| replace:: `train_result <df_t_api_result_>`_
.. |t_op| replace:: `train(...) <df_t_api_>`_

.. |i_math| replace:: `Inference <df_i_math_>`_
.. |i_dense| replace:: `Dense <df_i_math_dense_>`_
.. |i_hist| replace:: `Hist <df_i_math_hist_>`_
.. |i_input| replace:: `infer_input <df_i_api_input_>`_
.. |i_result| replace:: `infer_result <df_i_api_result_>`_
.. |i_op| replace:: `infer(...) <df_i_api_>`_

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_dense|     |t_hist|    |t_op|   |t_input|   |t_result|
   |i_math|        |i_dense|     |i_hist|    |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============

------------------------
Mathematical formulation
------------------------

.. _df_t_math:

Training
--------
Given :math:`n` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,x_n=(x_{n1},\ldots,x_{np})\}` of
size :math:`p`, their non-negative observation weights :math:`W=\{w_1,\ldots,w_n\}` and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\}`, 

    - :math:`y_i \in \{0, \ldots, C-1\}`, where :math:`C` is the number of classes, for classification
    - :math:`y_i \in \mathbb{R}`, for regression 

the problem is to build a decision forest classification or regression model.

During the training stage, :math:`B` independent classification or regression trees are created using the following:

#. New training set generated from the original one by sampling uniformly and with replacement
   (bootstrapping).
#. Impurity metric :math:`I` and impurity reduction :math:`\Delta I` for splitting tree's nodes, calculated as follows:
    - Gini impurity for classification:

      - without observation weights: :math:`I(D)=1-\sum_{i=1}^{C}{p_i^2},` where :math:`p_i` is the fraction of observations in subset :math:`D` that belong to the :math:`i`-th class.
      - with observation weights: :math:`I(D)=1-\sum_{i=1}^{C}{p_i^2},` where :math:`p_i` is the weighted fraction of observations in subset :math:`D` that belong to the :math:`i`-th class, computed as follows:
	   
        .. math::
           p_i=(\sum_{d \in \{d \in D | y_{d}=i\}}w_d)/\sum_{d \in D}w_d
        
        where :math:`w_d` is a weight of observation :math:`d`.

    - Mean-Square Error (MSE) for regression: 

      - without observation weights: :math:`I(D)=\frac{1}{N} \sum_{i=1}^{N}{(y_i - \bar{y})^2},` where :math:`N=|D|` and :math:`\bar{y}=\frac{1}{N} \sum_{i=1}^{N}y_i`.
      - with observation weights: :math:`I(D)=\frac{1}{W(D)} \sum_{i=1}^{N}w_i{(y_i - \bar{y})^2},` where :math:`N=|D|`, :math:`\bar{y}=\sum_{i=1}^{N}w_{i}y_{i},`, :math:`W(D)=\sum_{i=1}^{N}w_{i},` and :math:`w_i` is a weight of observation :math:`i`.
    - :math:`\Delta I` is computed as follows:
       
      .. math::
         \Delta I={I} - (\frac{N_{\mathrm{left}}}{N_{\mathrm{parent}}} I_{left} + \frac{N_{\mathrm{right}}}{N_{\mathrm{parent}}} I_{\mathrm{right}}) 
       
      where :math:`N_{\mathrm{left}}` and :math:`N_{\mathrm{right}}` are the number of observations in the node on the corresponding side of the split.

Let :math:`S=(X,Y)` be the set of observations. Given the training parameters, such as the number of trees
in the forest (:math:`B`), the fraction of observations used for the training of one tree
(:math:`f`), and the number of features to try as a possible split per
node (:math:`m`), the algorithm does the following:

#. For each tree (:math:`1, \ldots, B`):
#. Generate a bootstrapped set of observations with :math:`f * |S|`
   elements in it.
#. Start with the tree whose depth is equal to :math:`0`.
#. For each terminal node :math:`t` in the tree:
    - Choose randomly without replacement :math:`m` feature indices :math:`J_t \in \{0, 1, \ldots, p-1\}`.
    - For each :math:`j \in J_t`, find the best split :math:`s_{j,t}` that
      partitions subset :math:`D_t` and maximizes impurity decrease :math:`\Delta I_t`.
    - Get the best split :math:`s_t` that maximizes impurity decrease :math:`\Delta I_t` in all :math:`s_{j,t}` splits.
    - Split current node into two based the best split.
#. Stop when a termination criterion is met.

Termination Criteria
~~~~~~~~~~~~~~~~~~~~

The library supports the following termination criteria to stop growing the tree:
    - *Minimal number of observations in a leaf node*. Node :math:`t` is not processed if the subset of
      *observations* is smaller than the predefined value. Splits that produce nodes with the number of
      *observations* smaller than that value are not allowed.
    - *Maximal tree depth*. Node :math:`t` is not processed if its depth in the tree reaches the predefined
      maximal value.
    - *Impurity threshold*. Node :math:`t` is not processed if its :math:`I` value is smaller than the predefined threshold.

Random Numbers Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a *bootstrap* set and choose feature indices in the
performant way, the training algorithm requires the source of
random numbers, capable to produce sequences of random numbers in
parallel.

Initialization of the engine in the decision forest is based on
the scheme below:

The state of the engine is updated once the training of the
decision forest model is completed. The library provides support
to retrieve the instance of the engine with updated state that can
be used in other computations. The update of the state is
engine-specific and depends on the parallelization technique used
as defined earlier:

- Family: the updated state is the set of states that represent
  individual engines in the family.

- Leapfrog: the updated state is the state of the sequence with
  the rightmost position on the sequence. The example below
  demonstrates the idea for case of 2 subsequences (‘x’ and ‘o’)
  of the random number sequence:

- SkipAhead: the updated state is the state of the independent
  sequence with the rightmost position on the sequence. The
  example below demonstrates the idea for case of 2 subsequences
  (‘x’ and ‘o’) of the random number sequence:

Additional Characteristics Calculated by the Decision Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decision forests can produce additional characteristics, such as
an estimate of generalization error and an importance measure
(relative decisive power) of each of p features (variables).

Out-of-bag Error
++++++++++++++++

The estimate of the generalization error based on the training
data can be obtained and calculated as follows:

-  For classification:
    - For each vector :math:`x_i` in the dataset :math:`X`, predict its label :math:`\hat{y_i}` by having the majority of votes from the trees that
      contain :math:`x_i` in their OOB set, and vote for that label.

    - Calculate the OOB error of the decision forest :math:`T` as the average
      of misclassifications:

      .. math::
         OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{y_i \in {D}^{\text{'}}}I\{y_i \ne \hat{y_i}\}\text{,where }{D}^{\text{'}}={\bigcup }_{b=1}^{B}\overline{D_b}.

    - If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`: :math:`OOB(x_i) = I\{{y}_{i}\ne \hat{{y}_{i}}\}`

-  For regression:
    - For each vector :math:`x_i` in the dataset :math:`X`, predict its response :math:`\hat{y_i}`
      as the mean of prediction from the trees that contain :math:`x_i` in their OOB set:

      :math:`\hat{y_i} = \frac{1}{{|B}_{i}|}\sum _{b=1}^{|B_i|}\hat{y_{ib}}`, where :math:`B_i= \bigcup{T_b}: x_i \in \overline{D_b}` and :math:`\hat{y_{ib}}` is the result of prediction
      :math:`x_i` by :math:`T_b`.

    - Calculate the OOB error of the decision forest :math:`T` as the Mean-Square Error (MSE):

      .. math::
         OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{{y}_{i} \in {D}^{\text{'}}}\sum {(y_i-\hat{y_i})}^{2}, \text{where } {D}^{\text{'}}={\bigcup}_{b=1}^{B}\overline{{D}_{b}}

    - If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`:

      .. math::
         OOB(x_i) = {(y_i-\hat{y_i})}^{2}

Variable Importance
+++++++++++++++++++

There are two main types of variable importance measures:

- *Mean Decrease Impurity* importance (MDI).

  Importance of the :math:`j`-th variable for predicting :math:`Y` is the sum of
  weighted impurity decreases :math:`p(t) \Delta i(s_t, t)` for all nodes
  :math:`t` that use :math:`x_j`, averaged over all :math:`B` trees in the
  forest:

  .. math::
     MDI\left(j\right)=\frac{1}{B}\sum _{b=1}^{B} \sum _{t\in {T}_{b}:v\left({s}_{t}\right)=j}p\left(t\right)\Delta i\left({s}_{t},t\right),

  where :math:`p\left(t\right)=\frac{|{D}_{t}|}{|D|}` is the fraction of observations reaching node :math:`t`
  in the tree :math:`T_b`, and :math:`v(s_t)` is the index of the
  variable used in split :math:`s_t` .

- *Mean Decrease Accuracy* (MDA).

  Importance of the :math:`j`-th variable for predicting :math:`Y` is the average
  increase in the OOB error over all trees in the forest when the
  values of the :math:`j`-th variable are randomly permuted in the OOB
  set. For that reason, this latter measure is also known as
  *permutation importance*.

  In more details, the library calculates MDA importance as
  follows:

  - Let :math:`\pi (X,j)` be the set of feature vectors where the :math:`j`-th variable is randomly permuted over all vectors in the set.
  - Let :math:`E_b` be the OOB error calculated for :math:`T_b:` on its out-of-bag dataset :math:`\overline{D_b}`.
  - Let :math:`E_{b,j}` be the OOB error calculated for :math:`T_b:` using :math:`\pi \left(\overline{{X}_{b}},j\right)`, and its out-of-bag dataset :math:`\overline{D_b}` is permuted on the :math:`j`-th variable. Then

	* :math:`{\delta }_{b,j}={E}_{b}-{E}_{b,j}` is the OOB error increase for the tree :math:`T_b`.
	* :math:`Raw MDA\left(j\right)=\frac{1}{B}\sum _{b=1}^{B}{\delta }_{b,j}` is MDA importance.
	* :math:`Scaled MDA\left(j\right)=\frac{Raw MDA\left({x}_{j}\right)}{\frac{{\sigma }_{j}}{\sqrt{B}}}`, where :math:`{\sigma }_{j}^{2}` is the variance of :math:`D_{b,j}`

.. _df_t_math_dense:

Training method: *Dense*
------------------------
In *dense* training method all possible split variants for each feature (from selected features' subset for current node) are evaluated 
for best split computation.

.. _df_t_math_hist:

Training method: *Hist*
-----------------------
In *hist* training method, we consider only some selected subset of splits for best split computation. 
This subset of splits is computed for each feature on initialization stage of the algorithm. After computing subset of splits, we substitute 
each value from initially provided data with the value of the corresponding bin. Bins are continuous intervals between selected splits.

.. _df_i_math:
.. _df_i_math_dense:
.. _df_i_math_hist:

Inference methods: *Dense* and *Hist*
-------------------------------------
*Dense* and *hist* inference methods performs prediction by the same way:

#. For classification, :math:`y_i \in \{0, \ldots, C-1\}`, where :math:`C` is the number of classes,
   the tree ensemble model predicts the output by selecting the response :math:`y`,
   which is voted for by the majority of the trees in the forest.

#. For regression, the tree ensemble model uses the mean of :math:`B` functions' results to predict the
   output, i.e. :math:`\hat{y}=\frac{1}{M} \sum_{k=1}^M{f_k(x_i)}, \; f_k \in F,` where :math:`f_k` are
   regression trees, :math:`W` is a set of tree leaves' scores and :math:`T` is the number of leaves in the tree.
   In other words, each tree maps an observation to the corresponding leaf's score.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::decision_forest`` namespace and are available via inclusion of the
``oneapi/dal/algo/decision_forest.hpp`` header file.
                   
Enum classes
------------
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::error_metric_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::variable_importance_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::infer_mode
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::voting_mode
                   
Descriptor
----------
.. onedal_class:: oneapi::dal::decision_forest::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::task::v1

Model
-----
.. onedal_class:: oneapi::dal::decision_forest::v1::model

.. _df_t_api:

Training :expr:`train(...)`
--------------------------------
.. _df_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::train_input


.. _df_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::train_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::train_result train(const Descriptor& desc, \
                                         const decision_forest::train_input& input)

   :tparam Descriptor: Decision Forest algorithm descriptor :expr:`decision_forest::desc`.

   Preconditions
      | :expr:`input.data.is_empty == false`
      | :expr:`input.labels.is_empty == false`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.data.row_count == input.labels.row_count`
      | :expr:`desc.get_bootstrap() == true || 
               (desc.get_bootstrap() == false &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_raw &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_scaled)`
      | :expr:`desc.get_bootstrap() == true || 
               (desc.get_bootstrap() == false && desc.get_error_metric_mode() == error_metric_mode::none)`

.. _df_i_api:

Inference :expr:`infer(...)`
----------------------------
.. _df_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::infer_input


.. _df_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::infer_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::infer_result infer(const Descriptor& desc, \
                                         const decision_forest::infer_input& input)

   :tparam Descriptor: Decision Forest algorithm descriptor :expr:`decision_forest::desc`.

   Preconditions
      | :expr:`input.data.is_empty == false`
