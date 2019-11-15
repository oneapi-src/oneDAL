.. ******************************************************************************
.. * Copyright 2014-2019 Intel Corporation
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

.. _sycl-numeric-tables:

SYCL* Numeric Tables
====================

.. toctree::
   :maxdepth: 1
   :hidden:

``SyclNumericTable`` class is designed to allow user to hold processed
data on the device side and avoid extra data transfer between host and device.
If one tries to use traditional (i.e. no *SYCL\** in the name) numeric
tables with GPU algorithm, the data transfer will occur every
time the algorithm needs to get access to the data.

.. image:: ./images/traditional-numeric-table-flow.png
  :width: 600

Generally, a SYCL* numeric table is a wrapper around
regular SYCL* buffer acting as an adapter.
It enables the user to call GPU algorithms without
unnecessary data transfer.

.. image:: ./images/sycl-numeric-table-flow.png
  :width: 600

SYCL* Homogeneous Numeric Table
*******************************

For now, only ``SyclHomogenNumericTable`` is implemented for DPC++ interfaces. It has similar data layout as a traditional
``HomogenNumericTable`` and can be initialized, operated on and uninilialized like a traditional one.
Additional capabilities of ``SyclHomogenNumericTable`` are described in the sections below.

Initialize
----------

A SYCL* Homogeneous numeric table can be constructed in two ways:

- as a traditional ``HomogenNumericTable`` from host's CPU memory (see `Homogeneous Numeric Tables`_),
- by a one-dimentional ``cl::sycl::buffer``, in which case, the numeric table will hold a reference to the obtained SYCL* buffer.

.. code-block:: c++

    cl::sycl::buffer<float, 1> bf { data, cl::sycl::range<1>{ rows*cols } };
    // data is float* array with rows*cols elements

    // some operations...

    auto tablePtr = SyclHomogenNumericTable::create(bf, cols, rows);

Operate
-------

You can get underlying reference to SYCL* buffer from ``SyclHomogenNumericTable``
using ``getBlockOfRows()`` and ``getBlockOfColumns()`` methods.

.. code-block:: c++

    auto blockDescriptor = tablePtr->getBlockOfRows(0, rows, readWrite);
    // blockDescriptor is an object of BlockDescriptor<float> class

    auto readBuffer = blockDescriptor.getBuffer().toSycl();
    // readBuffer is cl::sycl::buffer<float, 1>

    // some operations with readBuffer ...

    tablePtr->releaseBlockOfRows(blockDescriptor);

.. note::
    ``SyclHomogenNumericTable`` does not own ``cl::sycl::buffer`` it was created from.
    Any changes of data into this buffer will affect the related numeric table. However, the buffer that you get
    from numeric table using ``getBlockOfRows()`` or ``getBlockOfColumns()`` methods may be another one.
    In this case, you should synchronize changes of data in buffer you got
    to numeric table by ``releaseBlockOfRows()`` method call.
