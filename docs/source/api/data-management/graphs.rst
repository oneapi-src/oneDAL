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

.. _api_graph_traits:

Graph traits
------------

Graph traits is a data type that defines the data model required for processing
and service functionality: a set of value types and ranges associated with graph
data types.  

graph_traits shall be specialized for each graph data structure.

::

   template <typename Graph>
   struct graph_traits {
      using graph_type = empty_value;
      using allocator_type = empty_value;

      // graph weight types
      using graph_user_value_type = empty_value;
      using const_graph_user_value_type = empty_value;

      // vertex types
      using vertex_type = empty_value;
      using vertex_iterator = empty_value;
      using const_vertex_iterator = empty_value;
      using vertex_size_type = empty_value;

      // vertex weight types
      using vertex_user_value_type = empty_value;

      // edge types
      using edge_type = empty_value;
      using edge_iterator = empty_value;
      using const_edge_iterator = empty_value;
      using edge_size_type = empty_value;

      // edge weight types
      using edge_user_value_type = empty_value;
   };

.. .. namespace:: oneapi::dal::preview
.. .. struct:: graph_traits

graph_traits<G>::graph_type
   Represents the type of the graph G.

graph_traits<G>::allocator_type
   Represents the type of allocator of the graph G.

graph_traits<G>::graph_user_value_type
   Represents the type of the :capterm:`attribute` of the graph G.

graph_traits<G>::const_graph_user_value_type
   Represents the constant type of the :capterm:`attribute` of the graph G.

graph_traits<G>::vertex_type
   Represents the type of vertices in the graph G.

graph_traits<G>::vertex_iterator
   Represents the type of the iterator by vertices in the graph G.

graph_traits<G>::const_vertex_iterator
   Represents the constant type of the iterator by vertices in the graph G.

graph_traits<G>::vertex_size_type
   Represents the type of the index of the vertices in the graph G.

graph_traits<G>::vertex_user_value_type
   Represents the type of the :capterm:`attributes <Attribute>` of the vertices
   in the graph G.

graph_traits<G>::edge_type
   Represents the type of edges in the graph G.

graph_traits<G>::edge_iterator
   Represents the type of the iterator by edges in the graph G.

graph_traits<G>::const_edge_iterator
   Represents the constant type of the iterator by edges in the graph G.

graph_traits<G>::edge_size_type
   Represents the type of the index of the edges in the graph G.

graph_traits<G>::edge_user_value_type
   Represents the type of the :capterm:`attributes <Attribute>` of the edges
   in the graph G.

.. _api_graph_service_funcs:

Graph service functions
-----------------------

Graph service functions is the set of functionality for access to the graph.



.. toctree::

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
