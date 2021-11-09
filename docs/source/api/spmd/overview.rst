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

.. highlight:: cpp
.. default-domain:: cpp

.. _api_spmd:

===============================
Distributed SPMD model overview
===============================

Refer to :ref:`Developer Guide: SPMD <spmd_overview>`.

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::spmd::preview`` namespace and are available via inclusion of the
header file from specified backend.

SPMD distributed model consists of the the following components:

1. Additional ``train``, ``infer``, and ``compute`` methods that accept ``communicator`` object
   as the first parameter. Those methods are expected to be called on all ranks to start
   distributed simulations.

2. The ``communicator`` class that contains methods to perform collective operations among all ranks.

3. Free functions to create a communicator using a specified backed. Available backends are ``ccl``
   and ``mpi``.

-------------
Usage example
-------------

The following listings provide a brief introduction how to create a particular communicator.
MPI backend:

.. include:: ../../includes/spmd/mpi-comm-usage-example.rst

.. rubric:: CCL backend

.. include:: ../../includes/spmd/ccl-comm-usage-example.rst
