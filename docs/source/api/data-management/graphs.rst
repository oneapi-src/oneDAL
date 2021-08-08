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

.. _api_graphs:

======
Graphs
======

Refer to :ref:`Developer Guide: Graphs <dm_graphs>`.

.. _graph_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::preview`` namespace and be available via inclusion of the
``oneapi/dal/graph/common.hpp`` header file.

.. _graph_traits:

Graph traits
------------

graph_traits defines the core data model required for the functions and 
algorithms in this proposal. It defines a set of value types and ranges for a graph data structure used by the functions and algorithms in this proposal. While usable as-is, they are typically used by the template type aliases, such as vertex_edge_range<G>, edge_user_value_type<G> and edge_iterator<G> in algorithms.

graph_traits needs to be specialized for each graph data structure.

.. toctree::

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
