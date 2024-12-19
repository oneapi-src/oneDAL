.. Copyright contributors to the oneDAL project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. highlight:: cpp

.. _dm_backend_primitives:

==================
Backend Primitives
==================

This section describes the types related to data management backend primitives.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Data Management Backend Primitives Types
   :header-rows: 1
   :widths: 10 70
   :class: longtable

   * - Type
     - Description

   * - :ref:`api_ndorder`
     - An enumeration of multidimensional data orders used to store
       contiguous data blocks inside the table.

   * - :ref:`api_ndshape`
     - A class that represents the shape of a multidimensional array.

   * - :ref:`api_ndview`
     - An implementation of a multidimensional data container that provides a view of the homogeneous
       data stored in an externally-managed memory block.

   * - :ref:`api_ndarray`
     - A class that provides a way to store and manipulate homogeneous data
       in a multidimensional structure.

---------------------
Programming interface
---------------------

Refer to :ref:`API: Data Management Backend Primitives <backend_primitives_programming_interface>`.
