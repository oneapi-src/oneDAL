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

::

   #include <CL/sycl.hpp>
   #include <iostream>
   #include <string>
   #include "oneapi/dal/array.hpp"

   using namespace oneapi;

   void print_property(const std::string& description, const auto& property) {
      std::cout << description << ": " << property << std::endl;
   }

   int main() {
      sycl::queue queue { sycl::default_selector() };

      constexpr std::int64_t data_count = 4;
      const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f };

      // Creating an array from immutable user-defined memory
      auto arr_data = dal::array<float>::wrap(data, data_count);

      // Creating an array from internally allocated memory filled by ones
      auto arr_ones = dal::array<float>::full(queue, data_count, 1.0f);

      print_property("Is arr_data mutable", arr_data.has_mutable_data()); // false
      print_property("Is arr_ones mutable", arr_ones.has_mutable_data()); // true

      // Creating new array from arr_data without data copy - they share ownership information.
      dal::array<float> arr_mdata = arr_data;

      print_property("arr_mdata elements count", arr_mdata.get_count()); // equal to data_count
      print_property("Is arr_mdata mutable", arr_mdata.has_mutable_data()); // false

      /// Copying data inside arr_mdata to new mutable memory block.
      /// arr_data still refers to the original data pointer.
      arr_mdata.need_mutable_data(queue);

      print_property("Is arr_data mutable", arr_data.has_mutable_data()); // false
      print_property("Is arr_mdata mutable", arr_mdata.has_mutable_data()); // true

      queue.submit([&](sycl::handler& cgh){
         auto mdata = arr_mdata.get_mutable_data();
         auto cones = arr_ones.get_data();
         cgh.parallel_for<class array_addition>(sycl::range<1>(data_count), [=](sycl::id<1> idx) {
            mdata[idx[0]] += cones[idx[0]];
         });
      }).wait();

      std::cout << "arr_mdata values: ";
      for(std::int64_t i = 0; i < arr_mdata.get_count(); i++) {
         std::cout << arr_mdata[i] << ", ";
      }
      std::cout << std::endl;

      return 0;
   }

.. _data_ownership_requirements:

---------------------------
Data ownership requirements
---------------------------

The array supports the following requirements on the internal data management:

1. An array owns two properties representing raw pointers to the data:

   - ``data`` for a pointer to immutable data block
   - ``mutable_data`` for a pointer to mutable data block (see the :txtref:`programming_interface`)

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

.. _programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal`` namespace and be available via inclusion of the
``oneapi/dal/array.hpp`` header file.

All the ``array`` class methods can be divided into several groups:

1. Constructors that are used to create an array from external, mutable or
   immutable memory.

2. Constructors and assignment operators that are used to create an array that shares its data
   with another one.

3. The group of ``reset()`` methods that are used to re-assign an array to another external
   memory block.

4. The group of ``reset()`` methods that are used to re-assign an array to an internally
   allocated memory block.

5. The methods that are used to access the data.

6. Static methods that provide simplified ways to create an array either from external
   memory or by allocating it within a new object.

.. onedal_class:: oneapi::dal::v1::array
