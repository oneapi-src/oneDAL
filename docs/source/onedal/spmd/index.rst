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

.. _dg_spmd:

============================
Single Program Multiple Data
============================

This section includes concepts and objects that support distributed simulation using SPMD model. 

#. Overview

In computing, SPMD (Single Program, Multiple Data) is a technique employed to achieve parallelism.
In SPMD, multiple autonomous processors simultaneously execute the same program at independent points.


#. Algorithm computation

In typical usage scenarios, a user provides a communicator object as a first parameter of a free function
to indicate that the algorithm can process data simultaneously. All internal inter-process communications
at sync points are hidden from the user. 


.. _typical_spmd_flow:

.. figure:: _static/spmd_flow.png
  :width: 800
  :alt: Typical SPMD flow

  Typical of SPMD Flow in oneDAL

Key concepts
============

.. spmd_communicators:

Communicator
------------

All collective operation between processes running simultaneously are enable
by a :capterm:`communicator`. It is a wrapper over specific backend (transport layer).

The communicator is used to perform collective operations on data and to obtain process
rank and total count of processes.

.. _communicator_backend:

Communicator backend
--------------------

Communicator backend is a underlying implementation of SPMD communicator object.
using specific library/interface:
- OneMPI
- OneCCL

.. _communicator_operations:

Supported Collective Operations
-------------------------------

The following collective operations are supported:

- Bcast
  Broadcasts data from specified process. 

- Allreduce
  Reduces data among all processes.

- Allgatherv
  Gather data from all processes and share the result among all processes.

.. _backend_restrictions:

Backend specific restrictions
-----------------------------

#. OneCCL:
Allgetherv doesn't support arbitrary displacements. The result is expected to 
be closely packed without gaps.

#. OneMPI:
Collective operations don't support asynchronous executions. All the ones block the
process till completion.
