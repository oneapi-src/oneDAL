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

Distributed Processing
======================

This mode assumes that the data set is split into ``nBlocks`` blocks across computation nodes.

To compute DBSCAN algorithm in the distributed processing mode, 
use the general schema described in Algorithms as follows:

- `Step 1 - on Local Nodes`_
- `Step 2 - on Local Nodes`_
- `Step 3 - on Local Nodes`_
- `Step 4 - on Local Nodes`_
- `Step 5 - on Local Nodes`_
- `Step 6 - on Local Nodes`_
- `Step 7 - on Master Node`_
- `Step 8 - on Local Nodes`_
- `Step 9 - on Master Node`_
- `Step 10 - on Local Nodes`_
- `Step 11 - on Local Nodes`_
- `Step 12 - on Local Nodes`_
- `Step 13 - on Local Nodes`_

Step 1 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-1.rst

Step 2 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-2.rst

Step 3 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-3.rst

Step 4 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-4.rst

Step 5 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-5.rst

Step 6 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-6.rst

Step 7 - on Master Node
***********************

.. include:: ./distributed-steps/step-7.rst

Step 8 - on Local Nodes
***********************

.. include:: ./distributed-steps/step-8.rst
    
Step 9 - on Master Node
***********************

.. include:: ./distributed-steps/step-9.rst

Step 10 - on Local Nodes
************************

.. include:: ./distributed-steps/step-10.rst

Step 11 - on Local Nodes
************************

.. include:: ./distributed-steps/step-11.rst

Step 12 - on Local Nodes
************************

.. include:: ./distributed-steps/step-12.rst

Step 13 - on Local Nodes
************************

.. include:: ./distributed-steps/step-13.rst

Examples
********

- C++: dbscan_dense_distr.cpp
- Python*: dbscan_batch.py
