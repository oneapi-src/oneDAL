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

Sorting
=======

In |short_name| sorting is an algorithm to sort the observations by each feature (column) in the ascending order.

The result of the sorting algorithm applied to the matrix 
:math:`X = (x_{ij})_{n \times p}` is the matrix :math:`Y = (y_{ij})_{n \times p}`
where the :math:`j`-th column :math:`(Y)_j = ( y_{ij} )`, :math:`i = 1, \ldots, n`,
is the column :math:`(X)_j = ( x_{ij} )`, :math:`i = 1, \ldots, n`, sorted in the ascending order.

Batch Processing
****************

At this moment, the description of batch processing for sorting is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-23>`_.
