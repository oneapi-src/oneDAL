.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
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

.. highlight:: cpp

.. _dm_graph_service:

=============
Graph Service
=============

This section defines :txtref:`requirements <graph_service_reqs>` to a
:txtref:`graph_service` implementation.

.. _graph_service_reqs:

------------
Requirements 
------------

Each function implementation from :txtref:`graph_service` concept:

1. Follows the definition of the :txtref:`graph_service` concept and its restrictions
   (e.g., read only access).

2. Is a free function (not method of :txtref:`graph` class type).

3. Uses :txtref:`graph_traits <api_graph_traits>` type referencing from input graph type.


-------------
Usage example
-------------

Example.

This section includes lists of service functions supported by specific :txtref:`graph` class type:


.. toctree::

   graph-service/undirected-adjacency-vector-graph.rst
   graph-service/directed-adjacency-vector-graph.rst