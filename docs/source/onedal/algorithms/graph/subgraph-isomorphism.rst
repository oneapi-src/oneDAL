.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. default-domain:: cpp

.. _alg_subgraph_isomorphism:

====================
Subgraph isomorphism
====================

.. include::  ../../../includes/graph/subgraph-isomorphism-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _subgraph_isomorphism_c_math:

Subgraphs definition
--------------------

A graph :math:`H = (V'; E')` is called a subgraph of graph :math:`G = (V; E)` if 
:math:`V' \subseteq V; E' \subseteq E` and :math:`V'` contains all the endpoints of all the 
edges in :math:`E'` [Gross2014]_.

Further we denote the :capterm:`induced subgraph on the vertex set<Induced subgraph on the vertex set>` 
as **induced** subgraph, the :capterm:`induced subgraph on the edge set<Induced subgraph on the edge set>`
as **non-induced** subgraph.

Computing
---------

Given two graphs :math:`G` and :math:`H`, the problem is to define whether graph :math:`G` contains
a subgraph which is isomorphic to graph :math:`H` and find exact mapping of subgraph :math:`H` in
graph :math:`G`.

Mapping is bijection or one-to-one correspondence between vertices of :math:`H` and subgraph of 
graph :math:`G` with correspond vertex adjacensy relationship preserving for non-induced subgraph 
and adjacensy and non-adjacensy preserving relationship for induced case. :math:`G` is called
**target** graph, :math:`H` is called **pattern** graph.


.. figure:: images/subgraph-isomorphism-target-pattern.png
    :width: 600
    :align: center
    :alt:

One of mappings for subgraph :math:`H` in graph :math:`G` in induced case is: [3, 0, 1, 4, 2, 5]. 
Which means that:

* pattern vertex with id 0 is mapped on vertex in target graph with id 3; 
* pattern vertex with id 1 is mapped on vertex in target graph with id 0;
* pattern vertex with id 2 is mapped on vertex in target graph with id 1;
* pattern vertex with id 3 is mapped on vertex in target graph with id 4.
* pattern vertex with id 4 is mapped on vertex in target graph with id 2.
* pattern vertex with id 5 is mapped on vertex in target graph with id 5.


Mapping for subgraph :math:`H` in graph :math:`G` in non-induced case is: [3, 0, 1, 4, 2, 5] and [3, 6, 1, 4, 2, 5].

.. _subgraph_isomorphism_c_dense:

Computation method: *dense*
---------------------------

The method defines VF3-light algorithms with Global State Stack parallelization method and supports induced and non-induced case.

For more details, see [Carletti2021]_.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Subgraph Isomorphism <api_subgraph_isomorphism>`.

--------
Examples
--------

.. include:: ../../../includes/graph/subgraph-isomorphism-examples.rst
