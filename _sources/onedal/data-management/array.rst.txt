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
.. default-domain:: cpp

.. _dm_array:

=====
Array
=====

The array is a simple concept over the data in |short_name|. It represents
a storage that:

1. Holds the data allocated inside it or references to the external data. The
   data are organized as one :term:`homogeneous <Homogeneous data>` and
   :term:`contiguous <Contiguous data>` memory block.

2. Contains information about the memory block's size.

3. Supports both :term:`immutable <Immutability>` and mutable data.

4. Provides an ability to change the data state from immutable to
   mutable one.

5. Holds ownership information on the data (see the :txtref:`data_ownership_requirements` section).

6. Ownership information on the data can be shared between several arrays. It is
   possible to create a new array from another one without any data copies.

-------------
Usage example
-------------

The following listing provides a brief introduction to the array API and an example of basic
usage scenario:

.. include:: ../../includes/data-management/array-usage-example.rst

.. _data_ownership_requirements:

---------------------------
Data ownership requirements
---------------------------

The array supports the following requirements on the internal data management:

1. An array owns two properties representing raw pointers to the data:

   - ``data`` for a pointer to immutable data block
   - ``mutable_data`` for a pointer to mutable data block (see the :txtref:`api_array`)

2. If an array owns mutable data, both properties point to the same memory
   block.

3. If an array owns immutable data, ``mutable_data`` is ``nullptr``.

4. An array stores the number of elements in the block it owns and updates
   the ``count`` property when a new memory block is assigned to the array.

5. An array stores a pointer to the **ownership structure** of the data:

   - The **reference count** indicating how many array objects refer to the
     same memory block.

   - The **deleter** object used to free the memory block when
     reference count is zero.

6. An array creates the ownership structure for a new memory block not
   associated with such structure.

7. An array decrements the number of references to the memory block when the
   array goes out of the scope. If the number of references is zero, the
   array calls the deleter on this memory block and free the ownership structure.

8. An array stores the pointer to the ownership structure created by another
   array when they share the data. An array increments the reference count
   for it to be equal to the number of array objects sharing the same data.

---------------------
Programming interface
---------------------

Refer to :ref:`API Reference: Array <api_array>`.