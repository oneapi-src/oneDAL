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

K-Means Clustering
==================

.. toctree::
   :glob:
   :maxdepth: 4



K-Means is among the most popular and simplest clustering methods. It
is intended to partition a data set into a small number of clusters
such that feature vectors within a cluster have greater similarity
with one another than with feature vectors from other clusters. Each
cluster is characterized by a representative point, called a
centroid, and a cluster radius.

In other words, the clustering methods enable reducing the problem of
analysis of the entire data set to the analysis of clusters.

There are numerous ways to define the measure of similarity and
centroids. For K-Means, the centroid is defined as the mean of
feature vectors within the cluster.


Details
*******

Given the set :math:`X = \{ x_1= (x_{11},\ldots{},x_{1p} ), \ldots{}, x_n = (x_{n1},\ldots{},x_{np} ) \}`
of :math:`n` :math:`p`-dimensional feature vectors and a positive integer :math:`k`, the
problem is to find a set :math:`C = \{ c_1, \ldots{} , c_k \}`
of :math:`k` :math:`p`-dimensional vectors that minimize the objective function
(overall error)

.. math::

	{\text{Φ}}_{X}(C)=\sum _{{x}_{i\in X}}{d}^{2}({x}_{i},C)


where :math:`{d}^{2}({x}_{i},C)` is the distance from :math:`x_i`
to the closest center in :math:`C`, such as the Euclidean distance.
The vectors :math:`c_1, \ldots{} , c_k` are called centroids. To start
computations, the algorithm requires initial values of centroids.

Centroid Initialization
-----------------------

Centroids initialization can be done using these methods:

- Choice of first :math:`k` feature vectors from the data set :math:`X`.

-
 Random choice of :math:`k` feature vectors from the data set using the
 following simple random sampling draw-by-draw algorithm. The
 algorithm does the following:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers.

 #.
	Resumes from step 1 until the set of centers
	reaches the desired size :math:`k`.

-
 K-Means++ algorithm [Arthur2007],
 which selects centers with the probability proportional to
 their contribution to the overall error :math:`{\text{Φ}}_{X}(C)`
 according to the following scheme:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers :math:`C`.

 #.
	For each feature vector :math:`x_i` in :math:`X` calculates its
	minimal distance :math:`{d}({x}_{i},C)` from the current set of
	centers :math:`C`.

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with the probability :math:`\frac{{d}^{2}({x}_{i},C)}{{\text{ Φ}}_{X}(C)}`.

 #.
	Resumes from step 2 until the set of centers
	:math:`C` reaches the desired size :math:`k`.

-
 Parallel K-Means++ algorithm [Bahmani2012]
 that does the following:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers :math:`C`.

 #.
	Repeats :math:`nRounds` times:

	a.
	   For each feature vector :math:`x_i` from :math:`X` calculates its
	   minimal distance :math:`{d}({x}_{i},C)` from the current set
	   of centers :math:`C`.

	b.
	   Chooses :math:`L = oversamplingFactor \cdot k` feature vectors :math:`x_i` from :math:`X` with the probability :math:`\frac{{d}^{2}({x}_{i},C)}{{\text{ Φ}}_{X}(C)}`.

	c.
	   Excludes :math:`x_i` vectors chosen in the previous step
	   from :math:`X` and adds them to the current set of centers :math:`C`.

 #.
	For :math:`c_i \in C` sets :math:`w_i` to the ratings, the
	number of points in :math:`X` closer to :math:`c_i` than to any other
	point in :math:`C`.

 #.
	Applies K-Means++ algorithm with weights :math:`w_i` to the
	points in :math:`C`, which means that the following probability is
	used in step:

	.. math::
	   \frac{{w}_{i}{d}^{2}({x}_{i},C)}{\text{ }{\sum }_{{x}_{iϵX}}{w}_{i}{d}^{2}({x}_{i},C)}


 The algorithm parameters define the number of candidates :math:`L`
 selected in each round and number of rounds:

 -

	Choose :math:`oversamplingFactor` to make :math:`L = O(k)`.

 -

	Choose nRounds as :math:`O(\log({\text{Φ}}_{X}(C)))`, where :math:`{\text{Φ}}_{X}(C)`
	is the estimation of the goal function when the first center
	is chosen.
	[Bahmani2012] recommends to set :math:`nRounds` to a constant value not greater
	than :math:`8`.


Computation
-----------

Computation of the goal function includes computation of the
Euclidean distance between vectors :math:`||x_j - m_i||`.
The algorithm uses the following modification of the Euclidean
distance between feature vectors :math:`a` and :math:`b`: :math:`d(a,b) = d_1(a,b) + d_2(a,b)`, where :math:`d_1` is computed
for continuous features as

.. math::
 {d}_{1}(a,b)=\sqrt{{\sum }_{k=1}^{{p}_{1}}{({a}_{k}-{b}_{k})}^{2}}

and :math:`d_2` is computed for binary categorical features as

.. math::
 {d}_{2}(a,b)=\gamma \sqrt{\sum _{k=1}^{{p}_{2}}{({a}_{k}-{b}_{k})}^{2}}

In these equations, :math:`\gamma` γ weighs the impact of binary categorical
features on the clustering, :math:`p_1` is the number of continuous
features, and :math:`p_2` is the number of binary categorical
features. Note that the algorithm does not support non-binary
categorical features.

The K-Means clustering algorithm computes centroids using Lloyd's
method [Lloyd82].
For each feature vector :math:`x_1, \ldots{} , x_k`, you can also
compute the index of the cluster that contains the feature vector.

In some cases, if no vectors are assigned to some clusters on a
particular iteration, the iteration produces an empty cluster. It
may occur due to bad initialization of centroids or the dataset
structure. In this case, the algorithm uses the following strategy
to replace the empty cluster centers and decrease the value of the
overall goal function:

-

 Feature vectors, most distant from their assigned centroids,
 are selected as the new cluster centers. Information about
 these vectors is gathered automatically during the algorithm
 execution.

-

 In the distributed processing mode, most distant vectors from
 the local nodes are computed (Step 1), stored in
 *PartialResult*, and collected on the master node (Step 2). For
 more details, see the *PartialResult* description at Step 1 [Tan2005]_.


Initialization
**************

The K-Means clustering algorithm requires initialization of
centroids as an explicit step. Initialization flow depends by the
computation mode. Skip this step if you already calculated initial
centroids.

Batch Processing
----------------


Input
+++++

Centroid initialization for K-Means clustering accepts the input
described below. Pass the Input ID as a parameter to the methods
that provide input for your algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the :math:`n \times p` numeric table with the data to be clustered. The input can be an object of any class derived from NumericTable.


Parameters
++++++++++

The following table lists parameters of centroid initialization
for K-Means clustering, which depend on the initialization method
parameter method.


.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1
   :align: left

   * - Parameter
     - method
     - Default Value
     - Description
   * - algorithmFPType
     - any
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - Not applicable
     - defaultDense
     - Available initialization methods for K-Means clustering:

       * defaultDense - uses first nClusters points as initial centroids
       * deterministicCSR - uses first nClusters points as initial centroids for data in a CSR numeric table
       * randomDense - uses random nClusters points as initial centroids
       * randomCSR - uses random nClusters points as initial centroids for data in a CSR numeric table
       * plusPlusDense - uses K-Means++ algorithm [Arthur2007]
       * plusPlusCSR - uses K-Means++ algorithm for data in a CSR numeric table
       * parallelPlusDense - uses parallel K-Means++ algorithm [Bahmani2012]
       * parallelPlusCSR - uses parallel K-Means++ algorithm for data in a CSR numeric table

       For more details, see the algorithm description.

   * - nClusters
     - any
     - Not applicable
     - The number of clusters. Required.
   * - nTrials
     - parallelPlusDense, parallelPlusCSR
     - :math:`1`
     - The number of trails to generate all clusters but the first initial cluster. For details, see [Arthur2007]_, section 5
   * - oversamplingFactor
     - parallelPlusDense, parallelPlusCSR
     - :math:`0.5`
     - A fraction of nClusters in each of nRounds of parallel K-Means++. L=nClusters*oversamplingFactor points are sampled in a round. For details, see [Bahmani2012]_, section 3.3.
   * - nRounds
     - parallelPlusDense, parallelPlusCSR
     - :math:`5`
     - The number of rounds for parallel K-Means++. (L*nRounds) must be greater than nClusters. For details, see [Bahmani2012]_, section 3.3.
   * - engine
     - any
     - SharePtr< engines:: mt19937:: Batch>()
     - Pointer to the random number generator engine that is used internally for random numbers generation.


Output
++++++

Centroid initialization for K-Means clustering calculates the
result described below. Pass the Result ID as a parameter to the
methods that access the results of your algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - centroids
     - Pointer to the :math:`nClusters \times p` numeric table with the cluster centroids. By default, this result is an object of the HomogenNumericTable class, but you can define the result as an object of any class derived from NumericTable except PackedTriangularMatrix, PackedSymmetricMatrix, and CSRNumericTable.


Computation
***********


Batch Processing
----------------

Algorithm Input
+++++++++++++++

The K-Means clustering algorithm accepts the input described
below. Pass the Input ID as a parameter to the methods that
provide input for your algorithm.


.. list-table::
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the :math:`n \times p` numeric table with the data to be clustered. The input can be an object of any class derived from NumericTable.
   * - inputCentroids
     - Pointer to the :math:`nClusters \times p` numeric table with the initial centroids. The input can be an object of any class derived from NumericTable.


Algorithm Parameters
++++++++++++++++++++

The K-Means clustering algorithm has the following parameters:

.. list-table::
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
     - Available computation methods for K-Means clustering:

        -  defaultDense - implementation of Lloyd's algorithm
        -  lloydCSR - implementation of Lloyd's algorithm for CSR numeric tables

   * - nClusters
     - Not applicable
     - The number of clusters. Required to initialize the algorithm.
   * - maxIterations
     - Not applicable
     - The number of iterations. Required to initialize the algorithm.
   * - accuracyThreshold
     - :math:`0.0`
     - The threshold for termination of the algorithm.
   * - gamma
     - :math:`1.0`
     - The weight to be used in distance calculation for binary categorical features.
   * - distanceType
     - euclidean
     - The measure of closeness between points (observations) being clustered. The only distance type supported so far is the Euclidian distance.
   * - assignFlag
     - true
     - A flag that enables computation of assignments, that is, assigning cluster indices to respective observations.


Algorithm Output
++++++++++++++++

The K-Means clustering algorithm calculates the result described
below. Pass the Result ID as a parameter to the methods that access
the results of your algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - centroids
     -
       Pointer to the :math:`nClusters \times p` numeric table with the cluster centroids.
       By default, this result is an object of the HomogenNumericTable class,
       but you can define the result as an object of any class derived from
       NumericTable except PackedTriangularMatrix, PackedSymmetricMatrix, and
       CSRNumericTable.
   * - assignments
     -
       Use when assignFlag=true. Pointer to the :math:`n \times 1` numeric table with
       assignments of cluster indices to feature vectors in the input data. By
       default, this result is an object of the HomogenNumericTable class, but
       you can define the result as an object of any class derived from
       NumericTable except PackedTriangularMatrix, PackedSymmetricMatrix, and
       CSRNumericTable.
   * - objectiveFunction
     -
       Pointer to the :math:`1 \times 1` numeric table with the value of the goal function.
       By default, this result is an object of the HomogenNumericTable class,
       but you can define the result as an object of any class derived from
       NumericTable except CSRNumericTable.
   * - nIterations
     -
       Pointer to the :math:`1 \times 1` numeric table with the actual number of iterations
       done by the algorithm. By default, this result is an object of the
       HomogenNumericTable class, but you can define the result as an object of
       any class derived from NumericTable except PackedTriangularMatrix,
       PackedSymmetricMatrix, and CSRNumericTable.


.. note::
  You can skip update of centroids and objectiveFunction in the
  result and compute assignments using original inputCentroids. To
  do this, set assignFlag to true and maxIterations to zero.

Examples
++++++++

C++:

- :cpp_example:`kmeans/kmeans_dense_batch.cpp`
- :cpp_example:`kmeans/kmeans_csr_batch.cpp`

Java*:

- :java_example:`kmeans/KMeansDenseBatch.java`
- :java_example:`kmeans/KMeansCSRBatch.java`

Performance Considerations
**************************

To get the best overall performance of the K-Means algorithm:

-
  If input data is homogeneous, provide the input data and store
  results in homogeneous numeric tables of the same type as
  specified in the algorithmFPType class template parameter.

-
  If input data is non-homogeneous, use AOS layout rather than SOA
  layout.

-
  For the output assignments table, use a homogeneous numeric table
  of the int type.



.. include:: ../../../opt-notice.rst

