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

.. _api_graph_service:

==============
Graph Service 
==============

Refer to :ref:`Developer Guide: Graph service <dm_graph_service>`.

.. _graph_service_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::preview`` namespace and be available via inclusion of the
``oneapi/dal/graph/service_functions.hpp`` header file.


The :ref:`graph service <graph_service>` is represented by a functionality
set providing a way to access the :txtref:`graph <api_graphs>`.


Each graph service function is a templated function with parameter ``Graph`` 
and uses :txtref:`aliases <graph_aliases>` on ``graph_traits`` for ``Graph`` types access.


.. _graph_aliases:


Related types
--------------

Aliases are used for shorter access to the :txtref:`graph <api_graph_template_type>` types.

.. list-table:: 
   :widths: 40 60 
   :header-rows: 1

   * - Alias
     - Value
   * - ``graph_allocator<G>``
     - ``graph_traits<G>::allocator_type``
   * - ``graph_user_value_type<G>``
     - ``graph_traits<G>::graph_user_value_type``
   * - ``vertex_user_value_type<G>``
     - ``graph_traits<G>::vertex_user_value_type``
   * - ``edge_user_value_type<G>``
     - ``graph_traits<G>::edge_user_value_type``
   * - ``vertex_type<G>``
     - ``graph_traits<G>::vertex_type``
   * - ``vertex_size_type<G>``
     - ``graph_traits<G>::vertex_size_type``
   * - ``edge_size_type<G>``
     - ``graph_traits<G>::edge_size_type``
   * - ``vertex_edge_size_type<G>``
     - ``graph_traits<G>::vertex_edge_size_type``
   * - ``vertex_outward_edge_size_type<G>``
     - ``graph_traits<G>::vertex_outward_edge_size_type`` 
   * - ``vertex_edge_iterator_type<G>``
     - ``graph_traits<G>::vertex_edge_iterator_type``
   * - ``const_vertex_edge_iterator_type<G>``
     - ``graph_traits<G>::const_vertex_edge_iterator_type``
   * - ``vertex_edge_range_type<G>``
     - ``graph_traits<G>::vertex_edge_range_type``
   * - ``const_vertex_edge_range_type<G>``
     - ``graph_traits<G>::const_vertex_edge_range_type``
   * - ``const_vertex_outward_edge_range_type<G>``
     - ``graph_traits<G>::const_vertex_outward_edge_range_type``    


Graph service functions
------------------------

Any service function has the following pattern:

.. code-block:: cpp

  template <typename Graph>
  some_type<Graph> graph_service_func(const Graph& g, ...);

.. .. namespace:: oneapi::dal::preview
.. .. func:: graph_service_func
 

.. _api_get_vertex_count: 

.. onedal_func:: oneapi::dal::preview::get_vertex_count 

.. _api_get_edge_count:

.. onedal_func:: oneapi::dal::preview::get_edge_count

.. _api_get_vertex_degree:

.. onedal_func:: oneapi::dal::preview::get_vertex_degree

.. _api_get_vertex_neighbors:

.. onedal_func:: oneapi::dal::preview::get_vertex_neighbors

.. _api_get_vertex_outward_degree:

.. onedal_func:: oneapi::dal::preview::get_vertex_outward_degree

.. _api_get_vertex_outward_neighbors:

.. onedal_func:: oneapi::dal::preview::get_vertex_outward_neighbors

.. _api_get_edge_value:

.. onedal_func:: oneapi::dal::preview::get_edge_value

Specified graphs service
-------------------------

This section contains description of service functions
which can access to the specified :txtref:`graph <api_graphs>` type.

.. toctree::

   graph-service/undirected-adjacency-vector-graph.rst
   graph-service/directed-adjacency-vector-graph.rst
