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

Service functins may or may not accesses specified graph type
due to the validity of the operation  on the related :txtref:`graph concept <graph_concepts>`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Service function
     - Valid :txtref:`graph concepts <graph_concepts>` 
   * - ``get_vertex_count``
     - :capterm:`undirected graph`, :capterm:`directed graph` 
   * - ``get_edge_count``
     - :capterm:`undirected graph`, :capterm:`directed graph`
   * - ``get_vertex_degree``
     - :capterm:`undirected graph`
   * - ``get_vertex_outward_degree``
     - :capterm:`directed graph`
   * - ``get_vertex_neighbors``
     - :capterm:`undirected graph`
   * - ``get_vertex_outward_neighbors``
     - :capterm:`directed graph`
   * - ``get_edge_value``
     - :capterm:`undirected graph`, :capterm:`directed graph`


.. _graph_service_reqs:

Requirements 
------------

Each function implementation from :txtref:`graph service functionality set<graph_service>`:

1. Has read only access.

2. Is a free function (not a method of :txtref:`graph` class type).

3. Uses :txtref:`graph_traits <api_graph_traits>` type referencing from input graph type.

-------------
Usage example
-------------

.. code-block:: cpp

  using graph_type = ...;
  const my_graph_type g = ...;
  std::cout << "Number of vertices: " << oneapi::dal::preview::get_vertex_count(g) << std::endl;
  std::cout << "Number of edges: " << oneapi::dal::preview::get_edge_count(g) << std::endl;

.. .. namespace:: oneapi::dal::preview
.. .. example:: graph_service_example

----------------
Lists of service
----------------

This section includes lists of service functions supported by specific :txtref:`graph` class type:

.. toctree::

   graph-service/undirected-adjacency-vector-graph.rst
   graph-service/directed-adjacency-vector-graph.rst