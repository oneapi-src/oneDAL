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

.. _dm_graphs:

======
Graphs
======

This section describes the types related to the :txtref:`graph` concept.

.. list-table::
   :header-rows: 1
   :widths: 10 70

   * - Type
     - Description

   * - :ref:`api_graphs_data_layout`
     - An enumeration of :capterm:`data layouts<data layout>` used to store
       adjacency matrix inside the graph.

---------------------------
Requirements on graph types
---------------------------

Each implementation of :txtref:`graph` concept:

1. Follows the definition of the :txtref:`graph` concept and its restrictions
   (e.g., :capterm:`immutability`).

3. Is :term:`reference-counted <Reference-counted object>`.

4. :txtref:`Graph traits` should be defined for each new :txtref:`graph` type.

The following listing provides an example of graph API to illustrate table kinds
and copy-assignment operation:

-----------
Graph types
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Graph type
     - Description

   * - :txtref:`undirected_adjacency_vector_graph`
     - A sparse undirected weighted or unweighted graph that contains :term:`CSR representation <CSR data>` of
        data.

   * - :txtref:`directed_adjacency_vector_graph`
     - A sparse directed weighted or unweighted graph that contains :term:`CSR representation <CSR data>` of
        data.

.. toctree::
   :hidden:

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
