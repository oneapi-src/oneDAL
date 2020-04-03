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

Classification Stump
====================

A Classification Decision Stump is a model that consists of a one-level decision tree
where the root is connected to terminal nodes (leaves) [Friedman2017]_.
The library only supports stumps with two leaves. 
Two methods of split criterion are available: gini and information gain. 
See :ref:`dt_classification` for details.

Batch Processing
****************

At this moment, the description of batch processing for Classification Stump is only available in
`Developer Guide for Intel(R) DAAL <https://software.intel.com/en-us/daal-programming-guide-batch-processing-123>`_.
