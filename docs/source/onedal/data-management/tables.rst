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

.. _dm_tables:

======
Tables
======

This section describes the types related to the :txtref:`table` concept.

.. list-table::
   :header-rows: 1
   :widths: 10 70

   * - Type
     - Description

   * - :txtref:`table <table_programming_interface>`
     - A common implementation of the table concept. Base class for
       other table types.

   * - :txtref:`table_metadata <metadata_programming_interface>`
     - An implementation of :txtref:`table_metadata` concept.

   * - :ref:`api_tables_data_layout`
     - An enumeration of :capterm:`data layouts<data layout>` used to store
       contiguous data blocks inside the table.

   * - :ref:`api_tables_feature_type`
     - An enumeration of :capterm:`feature` types used in |short_name| to
       define set of available operations onto the data.

---------------------------
Requirements on table types
---------------------------

Each implementation of :txtref:`table` concept:

1. Follows the definition of the :txtref:`table` concept and its restrictions
   (e.g., :capterm:`immutability`).

2. Is derived from the :cpp:expr:`oneapi::dal::table` class. The behavior of this class can be
   extended, but cannot be weaken.

3. Is :term:`reference-counted <Reference-counted object>`.

4. Every new :cpp:expr:`oneapi::dal::table` sub-type defines a unique id number - the "kind"
   that represents objects of that type in runtime.

The following listing provides an example of table API to illustrate table kinds
and copy-assignment operation:

.. code-block:: cpp

  using namespace onedal;

  // Creating homogen_table sub-type.
  dal::homogen_table table1 = homogen_table::wrap(queue, data_ptr, row_count, column_count);

  // table1 and table2 share the same data (no data copy is performed)
  dal::table table2 = table1;

  // Creating an empty table
  dal::table table3;

  std::cout << table1.get_kind()     == table2.get_kind() << std::endl; // true
  std::cout << homogen_table::kind() == table2.get_kind() << std::endl; // true
  std::cout << table2.get_kind()     == table3.get_kind() << std::endl; // false

  // Referring table3 to the table2.
  table3 = table2;
  std::cout << table2.get_kind() == table3.get_kind() << std::endl; // true

-----------
Table types
-----------

|short_name| defines a set of classes that implement the
:txtref:`table` concept for a specific data format:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Table type
     - Description

   * - :txtref:`homogen_table`
     - A dense table that contains :term:`contiguous <Contiguous data>`
       :term:`homogeneous <Homogeneous data>` data.

---------------------
Programming interface
---------------------

Refer to :ref:`API: Tables <table_programming_interface>`.

.. toctree::
   :hidden:

   table/homogen.rst
