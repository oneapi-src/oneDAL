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

Online Processing
*****************

.. note:: Online processing mode for Principal Component Analysis is not available on GPU.

Online processing computation mode assumes that data arrives in blocks :math:`i = 1, 2, 3, \ldots, \mathrm{nblocks}`.

PCA computation in the online processing mode follows the general computation schema for online processing described in :ref:`algorithms`.

.. contents::
    :local:
    :depth: 1

Algorithm Input
---------------

The PCA algorithm in the online processing mode accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. list-table::
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n_i \times p` numeric table that represents the current data block.
       The input can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

The PCA algorithm in the online processing mode has the following parameters, depending on the computation method parameter method:

.. list-table::
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left

   * - Parameter
     - Method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``defaultDense`` or ``svdDense``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - Not applicable
     - ``defaultDense``
     - Available computation methods for PCA computation:
     
       - ``defaultDense`` - the correlation method
       - ``svdDense`` - the SVD method
   * - ``initializationProcedure``
     - ``defaultDense`` or ``svdDense``
     - Not applicable
     - The procedure for setting initial parameters of the algorithm in the online processing mode.

       - By default, the algorithm with the ``defaultDense`` method initializes
         ``nObservationsCorrelation``, ``sumCorrelation``, and ``crossProductCorrelation`` with zeros.
       - By default, the algorithm with the ``svdDense`` method initializes
         ``nObservationsSVD``, ``sumSVD``, and ``sumSquaresSVD`` with zeros.

   * - ``covariance``
     - ``defaultDense``
     - `SharedPtr<covariance::Online<algorithmFPType, covariance::defaultDense> >`
     - The correlation and variance-covariance matrices algorithm to be used for PCA computations with the correlation method.
       For details, see :ref:`Correlation and Variance-covariance Matrices. Online Processing <cor_cov_online>`.

Partial Results
---------------

The PCA algorithm in the online processing mode calculates partial results described below.
They depend on the computation method.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabs::

    .. tab:: Correlation method (``defaultDense``)

        .. list-table::
            :widths: 10 60
            :header-rows: 1

            * - Result ID
              - Result
            * - ``nObservationsCorrelation``
              - Pointer to the :math:`1 \times 1` numeric table with the number of observations processed so far.
                
                .. note::
                        By default, this result is an object of the ``HomogenNumericTable`` class,
                        but you can define it as an object of any class derived from ``NumericTable`` except ``CSRNumericTable``.
            * - ``crossProductCorrelation``
              - Pointer to the :math:`p \times p` numeric table with the partial cross-product matrix computed so far. 

                .. note::

                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

            * - ``sumCorrelation``
              - Pointer to the :math:`1 \times p` numeric table with partial sums computed so far. 

                .. note::

                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.


    .. tab:: SVD method (``svdDense``)

        .. list-table::
            :widths: 10 60
            :header-rows: 1

            * - Result ID
              - Result
            * - ``nObservationsCorrelation``
              - Pointer to the :math:`1 \times 1` numeric table with the number of observations processed so far.
                
                .. note::
                        By default, this result is an object of the ``HomogenNumericTable`` class,
                        but you can define it as an object of any class derived from ``NumericTable`` except ``CSRNumericTable``.
            * - ``sumSVD``
              - Pointer to the :math:`1 \times p` numeric table with partial sums computed so far. 

                .. note::

                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

            * - ``sumSquaresSVD``
              - Pointer to the :math:`1 x p` numeric table with partial sums of squares computed so far.

                .. note::

                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Algorithm Output
----------------

The PCA algorithm in the online processing mode calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. list-table::
    :widths: 10 60
    :header-rows: 1

    * - Result ID
      - Result
    * - ``eigenvalues``
      - Pointer to the :math:`1 \times p` numeric table that contains eigenvalues in the descending order. 
    * - ``eigenvectors``
      - Pointer to the :math:`p \times p` numeric table that contains eigenvectors in the row-major order.
       
.. note::

    By default, these results are objects of the ``HomogenNumericTable`` class,
    but you can define the result as an object of any class derived from ``NumericTable``
    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
