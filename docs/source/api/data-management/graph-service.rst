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

============================
Graph service functionality
============================

Refer to :ref:`Developer Guide: Graph service <dm_graph_service>`.

.. _graph_service_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::preview`` namespace and be available via inclusion of the
``oneapi/dal/graph/service_functions.hpp`` header file.



Graph service 
--------------

The :ref:`graph service <graph_service>` concept is represented by a functionality set providing a way to access the :ref:`graph`.

Each graph service function is a templated function with parameter ``Graph`` and uses ``graph_traits`` for ``Graph`` types access.

In order to improve readability, the following aliases are introduced:

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
   * - ``vertex_type<G>``
     - ``graph_traits<G>::vertex_type``
   * - ``vertex_type<G>``
     - ``graph_traits<G>::vertex_type``

There are several graph service functions: 

.. onedal_func:: oneapi::dal::preview::get_vertex_count

api_2

api_3


.. toctree::

   graph-service/undirected-adjacency-vector-graph.rst
   graph-service/directed-adjacency-vector-graph.rst
