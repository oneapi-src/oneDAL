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

1. The instance stores a pointers to the graph :capterm:`topology`
   and :capterm:`attributes <Attribute>` of vertices and edges.

2. The reference count indicating how many graph objects refer to the same implementation.

3. The graph increments the reference count
   for it to be equal to the number of graph objects sharing the same implementation.

4. The graph decrements the reference count when the
   graph goes out of the scope. If the reference count is zero, the graph
   frees its implementation.

.. _api_graph_template_type:

The :txtref:`graph` types are defined as templated classes with the following parameters.

.. list-table:: 
   :widths: 30 70 
   :header-rows: 1

   * - Parameter
     - Description
   * - ``VertexValue``
     - Type of the vertices :capterm:`attributes <Attribute>`
   * - ``EdgeValue``
     - Type of the edge :capterm:`attributes <Attribute>`  
   * - ``GraphValue``
     - Type of the graph :capterm:`attributes <Attribute>`  
   * - ``IndexType``
     - Type of the vertex indices.
   * - ``Allocator``
     - An allocator type. Rebind is used to redefine for described above types. Allocator type shall be followed ``std::allocator_traits`` requirements.


:ref:`Empty value <api_empty_value>` tag structure shall be used as a value for 
``VertexValue``, ``EdgeValue``, ``GraphValue`` types if the corresponding
attribute is not set.

.. _api_empty_value:

.. onedal_class:: oneapi::dal::preview::empty_value

Graph class does not contain any public methods beside constructors 
and related overloaded ``operator=``. The graph is accessed  using the 
:ref:`service functions <api_graph_service>`.


.. _api_graph_traits:

Graph traits
------------
   
Graph traits is a data type that defines the data model, a set of types
associated with the graph, used by processing and service functionality.

Type graph_traits shall be specialized for each graph by following the pattern
below.

.. code-block:: cpp

   template <typename G>
   struct graph_traits {
      using graph_type = ...;
      using allocator_type = ...;
      ...
   };

.. .. namespace:: oneapi::dal::preview
.. .. struct:: graph_traits

The full list of types defined in ``graph_traits<G>`` is in the table below:

.. _graph_traits_types:

.. list-table:: 
   :widths: 20 40 20 20
   :header-rows: 1

   * - Type, ``graph_traits<G>::``
     - Description
     - :ref:`Undirected Adjacency Vector Graph <api_undirected_adjacency_vector_graph>`
     - :ref:`Directed Adjacency Vector Graph <api_directed_adjacency_vector_graph>`
   * - ``graph_type``
     - Type of the graph ``G``
     - ``undirected_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>``
     - ``directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>``
   * - ``allocator_type``
     - Type of the allocator of the graph ``G``
     - ``Allocator`` :ref:`[1] <GraphTemplateTypes>`
     - ``Allocator`` :ref:`[1] <GraphTemplateTypes>`
   * - ``graph_user_value_type``
     - Type of the :capterm:`attribute` of the graph ``G``
     - ``GraphValue`` :ref:`[1] <GraphTemplateTypes>`
     - ``GraphValue`` :ref:`[1] <GraphTemplateTypes>`
   * - ``const_graph_user_value_type``
     - Constant type of the :capterm:`attribute` of the graph ``G``
     - ``const GraphValue`` :ref:`[1] <GraphTemplateTypes>`
     - ``const GraphValue`` :ref:`[1] <GraphTemplateTypes>`
   * - ``vertex_type``
     - Type of the vertices in the graph ``G``
     - ``IndexType`` :ref:`[1] <GraphTemplateTypes>`
     - ``IndexType`` :ref:`[1] <GraphTemplateTypes>`
   * - ``vertex_iterator``
     - Type of the vertex iterator in the graph ``G``
     - ``vertex_type*``
     - ``vertex_type*``
   * - ``const_vertex_iterator``
     -  Constant type of the vertex iterator in the graph ``G``
     - ``const vertex_type*``
     - ``const vertex_type*``
   * - ``vertex_size_type``
     - Type of the index of the vertices in the graph ``G``
     - ``std::int64_t``
     - ``std::int64_t``
   * - ``vertex_user_value_type``
     - Type of the vertices :capterm:`attributes <Attribute>` of the graph ``G``
     - ``VertexValue`` :ref:`[1] <GraphTemplateTypes>`
     - ``VertexValue`` :ref:`[1] <GraphTemplateTypes>`
   * - ``edge_type``
     - Type of edges in the graph ``G``
     - ``std::int64_t``
     - ``std::int64_t``
   * - ``edge_iterator``
     - Type of the edge iterator in the graph ``G``
     - *Not available* 
     - *Not available*  
   * - ``const_edge_iterator``
     - Constant type of the edge iterator in the graph ``G``
     - *Not available*  
     - *Not available* 
   * - ``edge_size_type``
     - Type of the index of the edges in the graph ``G`` 
     - ``std::int64_t``
     - ``std::int64_t``
   * - ``edge_user_value_type``
     - Type of  edges :capterm:`attributes <Attribute>` 
     - ``EdgeValue`` :ref:`[1] <GraphTemplateTypes>`
     - ``EdgeValue`` :ref:`[1] <GraphTemplateTypes>`

.. _GraphTemplateTypes:     

[1] ``VertexValue``, ``EdgeValue``, ``GraphValue``, ``IndexType``, ``Allocator`` -- template parameters of graph G (see :ref:`example <api_directed_adjacency_vector_graph>`).
     
.. toctree::

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
