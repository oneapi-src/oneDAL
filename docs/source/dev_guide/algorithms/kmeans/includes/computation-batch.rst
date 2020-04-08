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
