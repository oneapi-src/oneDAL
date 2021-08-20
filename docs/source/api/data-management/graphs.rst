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

Graph
-----

The :txtref:`graph` concept is represented by types with suffix _graph and all of them are
:term:`reference-counted <Reference-counted object>`:

1. The instance stores a pointer to the graph :capterm:`topology`
   and :capterm:`attributes <Attribute>` of vertices and edges.

2. The reference count indicating how many graph objects refer to the same implementation.

3. The graph increments the reference count
   for it to be equal to the number of graph objects sharing the same implementation.

4. The graph decrements the reference count when the
   graph goes out of the scope. If the reference count is zero, the graph
   frees its implementation.

.. _graph_traits:

Graph traits
------------

Graph traits define the data model required for the functions and 
algorithms. It defines a set of value types and ranges for a graph data
structure used by the functions and algorithms. While usable as-is, they
are typically used by the template type aliases. Examples: vertex_edge_range<G>,
edge_user_value_type<G>, edge_iterator<G>.

graph_traits needs to be specialized for each graph data structure.

.. toctree::

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
