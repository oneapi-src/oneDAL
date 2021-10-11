.. ******************************************************************************
.. * Copyright 2014-2021 Intel Corporation
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

Build applications with oneDAL
==============================

See Get Started Guides for `Windows\*
<https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-daal-for-windows/top.html>`_
and `Linux\* <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-daal-for-linux/top.html>`_
for instruction on how to build applications for C++.
This section contains instructions for building applications with |short_name| for SYCL\*.

- :ref:`app_on_win`
- :ref:`app_on_lin`

.. _app_on_win:

Applications on Windows* OS
---------------------------

#. Download and install |base_tk|.

#. In Microsoft Visual Studio* Integrated Development Environment (IDE),
   open or create a C++ project for your |short_name| application to build.

#. In project properties:

   - Set |dpcpp| platform toolset:

     .. figure:: ./images/MSVSPlatformToolset.jpg
       :width: 600
       :align: center
       :alt: In General configuration properties, choose Platform Toolset property

   - Add |short_name| ``includes`` folder to :guilabel:`Additional Include Directories`.
   - Add folders with |short_name| and oneTBB libraries to :guilabel:`Library Directories`:

     .. figure:: ./images/LibraryDirectories.jpg
       :width: 600
       :align: center
       :alt: In VC++ Directories, choose Library Directories property

   - Add |short_name| and OpenCL libraries to :guilabel:`Additional Dependencies`:

     .. figure:: ./images/AdditionalDependencies.jpg
       :width: 600
       :align: center
       :alt: In Linker configuration properties, choose Input.

#. Add the appropriate libraries to your project based on |short_name| threading mode and linking method:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

   .. list-table:: |short_name| libraries for Windows
      :widths: 15 25 25
      :header-rows: 1
      :align: left
      :class: longtable

      * -
        - Single-threaded (non-threaded)
        - Multi-threaded (internally threaded)
      * - Static linking
        -
          | onedal_core.lib,
          | onedal_sequential.lib
        -
          | onedal_core.lib,
          | onedal_thread.lib
      * - Dynamic linking
        - onedal_core_dll.lib
        - onedal_core_dll.lib

   You may also add debug versions of the libraries based on the threading mode and linking method:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

   .. list-table:: |short_name| debug libraries for Windows
      :widths: 15 25 25
      :header-rows: 1
      :align: left
      :class: longtable

      * -
        - Single-threaded (non-threaded)
        - Multi-threaded (internally threaded)
      * - Static linking
        -
          | onedal_cored.lib,
          | onedald.lib,
          | onedal_dpcd.lib,
          | onedal_sycld.lib,
          | onedal_sequentiald.lib
        -
          | onedal_cored.lib,
          | onedald.lib,
          | onedal_dpcd.lib,
          | onedal_sycld.lib,
          | onedal_threadd.lib
      * - Dynamic linking
        -
          | onedal_cored_dll.lib (onedal_cored_dll.1.lib),
          | onedald_dll.lib (onedald_dll.1.lib),
          | onedal_dpcd_dll.lib (onedal_dpcd_dll.1.lib),
          | onedald.1.dll,
          | onedal_cored.1.dll,
          | onedal_dpcd.1.dll,
          | onedal_sequentiald.1.dll
        -
          | onedal_cored_dll.lib (onedal_cored_dll.1.lib),
          | onedald_dll.lib (onedald_dll.1.lib),
          | onedal_dpcd_dll.lib (onedal_dpcd_dll.1.lib),
          | onedald.1.dll,
          | onedal_cored.1.dll,
          | onedal_dpcd.1.dll,
          | onedal_threadd.1.dll

.. _app_on_lin:

Applications on Linux* OS
-------------------------

#. Download and install |base_tk|.

#. Set environment variables by calling ``<install dir>/setvars.sh``.

#. Build your application with clang++:

   - Add ``fsycl`` option to the command:

     .. code-block:: text

       -fsycl

   - Add ``ONEAPI_DAAL_USE_MKL_GPU_GEMM`` definition:

     .. code-block:: text

        -DONEAPI_DAAL_USE_MKL_GPU_GEMM

   - Add |short_name| ``includes`` folder:

     .. code-block:: text

        -I<install dir>/daal/latest/include

   - Add |short_name| libraries. Choose the appropriate |short_name| libraries based on |short_name| threading mode and linking method:

     .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

     .. list-table:: |short_name| libraries for Linux
          :widths: 15 25 25
          :header-rows: 1
          :align: left
          :class: longtable

          * -
            - Single-threaded (non-threaded)
            - Multi-threaded (internally threaded)
          * - Static linking
            -
              | libonedal_core.a,
              | libonedal_sequential.a
            -
              | libonedal_core.a,
              | libonedal_thread.a
          * - Dynamic linking
            -
              | libonedal_core.so,
              | libonedal_sequential.so
            -
              | libonedal_core.so,
              | libonedal_thread.so

   - Add an additional |short_name| library:

     .. code-block:: text

        -foffload-static-lib=<install dir>/daal/latest/libintel64/libonedal_sycl.a

Examples
********

Dynamic linking, Multi-threaded |short_name|:

.. code-block:: text

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group -L<install dir>/daal/latest/lib/intel64 -lonedal_core -lonedal_thread.so -lpthread -ldl -lOpenCL -L<install dir>/tbb/latest/lib/intel64/gcc4.8 -ltbb -ltbbmalloc -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libonedal_sycl.a -Wl,--end-group

Static linking, Single-threaded |short_name|:

.. code-block:: text

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group <install dir>/daal/latest/lib/intel64/libonedal_core.a <install dir>/daal/latest/lib/intel64/libonedal_sequential.a -lpthread -ldl -lOpenCL -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libonedal_sycl.a -Wl,--end-group
