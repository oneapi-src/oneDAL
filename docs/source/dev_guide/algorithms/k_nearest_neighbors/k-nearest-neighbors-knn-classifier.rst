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

k-Nearest Neighbors (kNN) Classifier
************************************

.. toctree::
   :glob:
   :maxdepth: 4


k-Nearest Neighbors (kNN) classification is a non-parametric
classification algorithm. The model of the kNN classifier is based on
feature vectors and class labels from the training data set. This
classifier induces the class of the query vector from the labels of
the feature vectors in the training data set to which the query
vector is similar. A similarity between feature vectors is determined
by the type of distance (for example, Euclidian) in a
multidimensional feature space.

Details
=======

The library provides kNN classification based on multidimensional
binary search tree (K-D tree, where D means the dimension and K means
the number of dimensions in the feature space). For more details, see
[James2013]_, [Patwary2016]_.

Given n feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p` and a vector
of class labels :math:`y = (y_1, \ldots, y_n)`, where :math:`y_i \in \{0, 1, \ldots, C-1\}`
and :math:`C` is the number of classes, describes the
class to which the feature vector :math:`x_i` belongs, the problem is
to build a kNN classifier.

Given a positive integer parameter :math:`k` and a test observation
:math:`x_0`, the kNN classifier does the following:

#. Identifies the set :math:`N_0` of the k feature vectors in the
   training data that are closest to :math:`x_0` according to the
   Euclidian distance
#. Estimates the conditional probability for the class :math:`j` as the
   fraction of vectors in :math:`N_0` whose labels y are equal to :math:`j`
#. Assigns the class with the largest probability to the test
   observation :math:`x_0`

|product| version of the kNN algorithm uses the PANDA algorithm
[Patwary2016]
that uses a space-partitioning data structure known as K-D tree. Each
non-leaf node of a tree contains the identifier of a feature along
which to split the feature space and an appropriate feature value (a
cut-point) that defines the splitting hyperplane to partition the
feature space into two parts. Each leaf node of the tree has an
associated subset (a bucket) of elements of the training data set.
Feature vectors from any bucket belong to the region of the space
defined by tree nodes on the path from the root node to the
respective leaf.

Training Stage
--------------

For each non-leaf node, the process of building a K-D tree
involves the choice of the feature (that is, dimension in the
feature space) and the value for this feature (a cut-point) to
split the feature space. This procedure starts with the entire
feature space for the root node of the tree, and for every next
level of the tree deals with ever smaller part of the feature
space.

The PANDA algorithm constructs the K-D tree by choosing the
dimension with the maximum variance for splitting
[Patwary2016].
Therefore, for each new non-leaf node of the tree, the algorithm
computes the variance of values that belong to the respective
region of the space for each of the features and chooses the
feature with the largest variance. Due to high computational cost
of this operation, PANDA uses a subset of feature values to
compute the variance.

PANDA uses a sampling heuristic to estimate the data distribution
for the chosen feature and chooses the median estimate as the
cut-point.

PANDA generates new K-D tree levels until the number of feature
vectors in a leaf node gets less or equal to a predefined
threshold. Once the threshold is reached, PANDA stops growing the
tree and associates the feature vectors with the bucket of the
respective leaf node.

Prediction Stage
----------------

Given kNN classifier and query vectors :math:`x_0, \ldots, x_r`,
the problem is to calculate the labels for those
vectors. To solve the problem for each given query vector
:math:`x_i`, the algorithm traverses the K-D tree to find feature
vectors associated with a leaf node that are closest to
:math:`x_i`. During the search, the algorithm limits exploration
of the nodes for which the distance between the query vector and
respective part of the feature space is not less than the distance
from the :math:`k^{th}` neighbor. This distance is progressively
updated during the tree traverse.


Batch Processing
================

kNN classification follows the general workflow described in
:ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the training stage, K-D tree based kNN classifier has the
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
     - The computation method used by the K-D tree based kNN classification.
       The only training method supported so far is the default dense method.
   * - nClasses
     - :math:`2`
     - The number of classes.
   * - dataUseInModel
     - doNotUse
     - A parameter to enable/disable use of the input data set in the kNN
       model. Possible values:

        + doNotUse - the algorithm does not include the input data and labels
          in the trained kNN model but creates a copy of the input data set.
        + doUse - the algorithm includes the input data and labels in the trained kNN model.

       The algorithm reorders feature vectors and corresponding labels in the
       input data set or its copy to improve performance at the prediction stage.

       If the value is doUse, do not deallocate the memory for input data and labels.

   * - engine
     - SharePtr< engines:: mt19937:: Batch>()
     - Pointer to the random number generator engine that is used internally to
       perform sampling needed to choose dimensions and cut-points for the K-D tree.

Prediction
----------

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the prediction stage, K-D tree based kNN classifier has the
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
     - The computation method used by the K-D tree based kNN classification.
       The only prediction method supported so far is the default dense method.
   * - nClasses
     - :math:`2`
     - The number of classes.
   * - :math:`k`
     - :math:`1`
     - The number of neighbors.

Examples
========

.. tabs::

  .. tab:: DPC++

    Batch Processing:

    - :ref:`bf_knn_dense_batch.cpp`
  
  .. tab:: C++

    Batch Processing:

    - :cpp_example:`kdtree_knn_dense_batch.cpp <k_nearest_neighbors/kdtree_knn_dense_batch.cpp>`

  .. tab:: Java*

    Batch Processing:

    - :java_example:`KDTreeKNNDenseBatch.java <kdtree_knn_classification/KDTreeKNNDenseBatch.java>`


.. Python*: kdtree_knn_dense_batch.py
