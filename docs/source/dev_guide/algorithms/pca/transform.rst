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

Principal Components Analysis Transform
=======================================

The PCA transform algorithm transforms the data set to principal components.

Details
*******

Given a transformation matrix :math:`T` computed by :ref:`PCA <pca>` (eigenvectors in row-major order)
and data set :math:`X` as input, the PCA Transform algorithm transforms input data set :math:`X`
of size :math:`n \times p` to the data set :math:`Y` of size :math:`n x p_r`, :math:`pr \leq p`.

Batch Processing
****************

At this moment, the description of batch processing for PCA Transform is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-8>`_.

DPC++ Examples
**************

Batch Processing:

- :ref:`pca_transform_dense_batch.cpp`
