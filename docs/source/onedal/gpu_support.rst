.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
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

.. _onedal_gpu_support:

CPU and GPU Support
===================
Not all computation modes, methods, and parameters are supported on both CPU and GPU.
Differences in CPU and GPU support are listed below.

GPU supported targets
*********************
oneDAL library targets Intel GPUs support only, while technically it should be feasible
to run on other HW platforms assuming SYCL runtime availability for platform.



TODO: below need to be updated in P
Methods
*******

For the following algorithms, only listed methods are supported on GPU:

.. tabularcolumns::  |\Y{0.5}|\Y{0.5}|

.. list-table:: GPU Support: Methods
   :header-rows: 1
   :widths: 10 10
   :align: left
   :class: longtable

   * - Algorithm
     - Supported on GPU
   * - :ref:`kmeans`
     - ``defaultDense``
   * - :ref:`kmeans_init`
     - ``defaultDense``, ``randomDense``
   * - /:ref:`linear_regression`
     - ``defaultDense``
   * - :ref:`moments_low_order`
     - ``defaultDense``
   * - :ref:`sgd_solver`
     - ``miniBatch``
   * - :ref:`Covariance <cor_cov>`
     - ``defaultDense``
   * - :ref:`pca`
     - ``defaultDense``
   * - :ref:`knn`
     - Brute Force
   * - :ref:`svm`
     - ``thunder``
   * - :ref:`decision_forest`
     - ``hist``

Parameters
**********

.. tabularcolumns::  |\Y{0.5}|\Y{0.5}|

.. list-table:: GPU Support: Algorithm Parameters
   :header-rows: 1
   :widths: 10 10
   :align: left
   :class: longtable

   * - Algorithm
     - Notes
   * - :ref:`svm`
     - ``doShrinking`` is only supported for ``defaultDense`` method.
   * - :ref:`dbscan`
     -
       - On GPU, the ``memorySavingMode`` flag can only be set to ``true``.
       - On GPU, the ``weights`` parameter is not supported.
   * - :ref:`kernel`
     - On GPU, the only supported computation mode (``ComputationMode``) is ``matrixMatrix``.
   * - :ref:`objective_function`
     -
       - On GPU, only :ref:`logistic_loss` and :ref:`cross_entropy_loss` are supported, :ref:`mse` is not supported.
       - On GPU, ``resultsToCompute`` only computes ``value``, ``gradient``, and ``hessian``.
   * - :ref:`logistic_regression`
     - ``penaltyL1`` is not supported on GPU
