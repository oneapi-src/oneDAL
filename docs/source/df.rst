.. default-domain:: cpp

.. _alg_df:

===================================
Decision Forest Classification and Regression (DF)
===================================

Decision Forest (DF) :capterm:`classification` and :capterm:`regression` algorithms are based on an ensemble of 
tree-structured classifiers, which are known as decision trees. Decision forest is built 
using the general technique of bagging, a bootstrap aggregation, and a random choice of features. Decision tree is
a binary tree graph. Its internal (split) nodes represent a decision function used to select the child node at the 
prediction stage. Its leaf, or terminal, nodes represent the corresponding response values, which are the result 
of the prediction from the tree. For more details, see Classification and Regression > Decision Tree, [Breiman84]_ and [Breiman2001]_.

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
size :math:`p` and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\}`, the problem is to build a decision forest
classification or regression model.

During the training stage, :math:`M` independent classification or regression trees are created using the following:

#. New training set generated from the original one by sampling uniformly and with replacement
   (bootstrapping).
#. Impurity metric (:math:`I`) and impurity reduction
   (:math:`I_{\mathrm{reduction}}`) for splitting tree's nodes, calculated as follows:
    - Gini impurity for classification:
        - without sample weights: :math:`I(D)=1-\sum_{i=1}^{C}{p_i^2},` where :math:`p_i` is the fraction of observations in subset D that belong to the :math:`i`-th class.
        - with sample weights: :math:`I(D)=1-\sum_{i=1}^{C}{p_i^2},` where :math:`p_i` is the weighted fraction of observations in subset D that belong to the :math:`i`-th class.
	  :math:`p_i=(\sum_{d \in \{d \in D | y_{d}=i\}}w_d)/\sum_{d \in D}w_d,` where :math:`w_d` is a weight of sample :math:`d`. 
    - Mean-Square Error (MSE) for regression: 
	- without sample weights: :math:`I(D)=1/N\sum_{i=1}^{N}{(y_i - \bar{y})^2},` where :math:`N=|D|` and :math:`\bar{y}=1/N\sum_{i=1}^{N}y_i`.
	- with sample weights: :math:`I(D)=1/W(D)\sum_{i=1}^{N}w_i{(y_i - \bar{y})^2},` where :math:`N=|D|` and :math:`\bar{y}=\sum_{i=1}^{N}w_{i}y_{i},` where :math:`W(D)=\sum_{i=1}^{N}w_{i},` and :math:`w_i` is a weight of sample :math:`i`.
    - :math:`I_{\mathrm{reduction}}={I} - (N_{\mathrm{left}}/N_{\mathrm{parent}}*I_{left} + N_{\mathrm{right}}/N_{\mathrm{parent}}*I_{\mathrm{right}}),` where :math:`N_{\mathrm{left}}` and :math:`N_{\mathrm{right}}` are the number of observations in the node on the corresponding side of the split.

Due to each tree being fitted to a bootstrapped subset of observations, those that are not used to
fit a given tree are referred to as out-of-bag (OOB) observations. We can calculate OOB error as the
average prediction error on each training sample :math:`x_i`, using only the trees that did not have :math:`x_i`
in their bootstrap sample.

Mean Decrease Impurity (feature importance MDI) metric is calculated as the sum of the impurity
reduction over all the nodes that are split on that feature (over all trees), averaged by a number
of trees.

Let :math:`S=(X,Y)` be the set of observations. Given the training parameters, such as the number of trees
in the forest (:math:`\mathrm{nTrees}`), the fraction of observations used for the training of one tree
(:math:`\mathrm{observationsPerTreeFraction}`), and the number of features to try as a possible split per
node (:math:`\mathrm{featuresPerNode}`), the algorithm does the following:

#. For each tree (:math:`1, \ldots, \mathrm{nTrees}`):
#. Generate a bootstrapped set of observations with :math:`\mathrm{observationsPerTreeFraction} * |S|`
   elements in it.
#. Start with the tree whose depth is equal to :math:`0`.
#. For each leaf node in the tree:
    - Randomly choose :math:`\mathrm{featuresPerNode}` features to be tried as possible splits.
    - Find the best split that maximizes :math:`{I}_{\mathrm{reduction}}`.
    - Split current node into two based the best split.
#. Stop when a termination criterion is met.
#. Update OOB error metric (optional).
#. Update feature importance metric (optional).

The library supports the following termination criteria to stop growing the tree:

- Minimal number of observations in a leaf node. Node :math:`t` is not processed if the subset of
  observations is smaller than the predefined value. Splits that produce nodes with the number of
  observations smaller than that value are not allowed.
- Maximal tree depth. Node :math:`t` is not processed if its depth in the tree reaches the predefined
  maximal value.
- Impurity threshold. Node :math:`t` is not processed if its :math:`I` value is smaller than the predefined
  minimal value defined with parameter :math:`\mathrm{impurityThreshold}`.

.. _df_t_math_dense:

Training method: *Dense*
~~~~~~~~~~~~~~~~~~~~~~~~
In "Dense" training method all possible split variants for each feature (from selected features' subset for current node) are evaluated 
for best split computation.

.. _df_t_math_hist:

Training method: *Hist*
~~~~~~~~~~~~~~~~~~~~~~~~
"inexact" (also called "histogram") training method. In this method we consider only some selected subset of splits for best split computation. 
This subset of splits is computed for each feature on initialization stage of the algorithm. After computing subset of splits, we substitute 
each value from initially provided data with the value of the corresponding bin. Bins are continuous intervals between selected splits.

.. _df_i_math:
.. _df_i_math_dense:
.. _df_i_math_hist:

Inference methods: *Dense* and *Hist*
-------------------------------------
"Dense" and "Hist" inference methods performs prediction by the same way:

#. For classification, :math:`y_i \in \{0, \ldots, \mathrm{C-1}\}`, where :math:`C` is the number of classes,
   the tree ensemble model predicts the output by selecting the response :math:`y`,
   which is voted for by the majority of the trees in the forest.

#. For regression, the tree ensemble model uses the mean of :math:`M` functions' results to predict the
   output, i.e. :math:`\hat{y}=1/M\sum_{k=1}^M{f_k(x_i)}, \; f_k \in F,` where :math:`F=\{f : \mathbb{R}^p \rightarrow W, \; W \subset \mathbb{R}, \; |W|=T\}` is a set of 
   regression trees, :math:`W` is a set of tree leaves' scores and :math:`T` is the number of leaves in the tree.
   In other words, each tree maps an observation to the corresponding leaf's score.

---------------------
Programming Interface
---------------------
All types and functions in this section shall be declared in the
``oneapi::dal::decision_forest`` namespace and be available via inclusion of the
``oneapi/dal/algo/decision_forest.hpp`` header file.
                   
Enum classes
------------
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::error_metric_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::variable_importance_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::infer_mode
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::voting_mode
                   
Descriptor
----------
.. onedal_class:: oneapi::dal::decision_forest::detail::v1::descriptor_base
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
.. onedal_func:: oneapi::dal::v1::train

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
.. onedal_func:: oneapi::dal::v1::infer
