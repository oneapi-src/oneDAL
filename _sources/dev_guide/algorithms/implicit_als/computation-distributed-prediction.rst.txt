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

.. _implicit_als_distributed_prediction:

Distributed Processing: Prediction of Ratings
=============================================

The distributed processing mode assumes that the data set is split in ``nblocks`` blocks across computation nodes.

.. contents::
    :local:
    :depth: 1


Algorithm Parameters
********************

At the prediction stage, implicit ALS recommender in the distributed processing mode has the following parameters:

.. list-table::
   :widths: 10 20 30
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - ``computeStep``
     - Not applicable
     - The parameter required to initialize the algorithm. Can be:
       
       - ``step1Local`` - the first step, performed on local nodes
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``nFactors``
     - :math:`10`
     - The total number of factors.

Use the one-step computation schema for implicit ALS recommender prediction in the distributed processing mode,
as explained below and illustrated by the graphic for :math:`\mathrm{nblocks} = 3`:

Step 1 - on Local Nodes
***********************

Prediction of rating uses partial models, which contain the parts of user factors :math:`X_1, X_2, \ldots, X_{\mathrm{nblocks}}`
and item factors :math:`Y_1, Y_2, \ldots, Y_{\mathrm{nblocks}}` produced at the training stage.
Each pair of partial models :math:`(X_i , Y_j)` is used to compute a numeric table with ratings :math:`R_{ij}`
that correspond to the user factors and item factors from the input partial models.

.. image:: images/implicit-als-distributed-computation-prediction-step-1.png
    :width: 800
    :align: center
    
In this step, implicit ALS recommender-based prediction accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. list-table::
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``usersPartialModel``
     - The partial model trained by the implicit ALS algorithm in the distributed processing mode.
       Stores user factors that correspond to the :math:`i`-th data block.
   * - ``itemsPartialModel``
     - The partial model trained by the implicit ALS algorithm in the distributed processing mode.
       Stores item factors that correspond to the :math:`j`-th data block.

In this step, implicit ALS recommender-based prediction calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. list-table::
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``prediction``
     - Pointer to the :math:`m_i \times n_j` numeric table with predicted ratings.
     
       .. note::
            By default this table is an object of the ``HomogenNumericTable`` class,
            but you can define it as an object of any class derived from ``NumericTable``
            except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
