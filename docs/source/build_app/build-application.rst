.. ******************************************************************************
.. * Copyright 2014-2020 Intel Corporation
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

See Get Started Guides for `Windows\* <https://software.intel.com/en-us/get-started-with-daal-for-windows>`_ 
and `Linux\* <https://software.intel.com/en-us/get-started-with-daal-for-linux>`_ 
for instruction on how to build applications for C++.
This section contains instructions for building applications with |short_name| for SYCL\*.

.. note::
  If you encounter a problem while building an application with |short_name|,
  refer to the list of :ref:`known issues <known_issues>`.

- `Applications on Windows`_
- `Applications on Linux`_

Applications on Windows
-----------------------

#. Download and install |base_tk|.

#. In Microsoft Visual Studio* Integrated Development environment (IDE), 
   open or create a C++ project for your |short_name| application to build.

#. In project properties set |dpcpp| platform toolset:

   .. image:: ./images/MSVSPlatformToolset.jpg
     :width: 600
     :align: center

#. In project properties add |short_name| ``includes`` folder to :guilabel:`Additional Include Directories`:


   .. image:: ./images/AdditionalIncludeDirs.jpg
     :width: 600
     :align: center
    

#. In project properties add folders with |short_name| and TBB libraries to :guilabel:`Library Directories`:

   .. image:: ./images/LibraryDirectories.jpg
     :width: 600
     :align: center

#. In project properties add |short_name| and OpenCL libraries to :guilabel:`Additional Dependencies`:

   .. image:: ./images/AdditionalDependencies.jpg
     :width: 600
     :align: center

#. Add the appropriate libraries to your project based on |short_name| threading mode and linking method:

     .. list-table::
          :widths: 25 25 25
          :header-rows: 1
          :align: left

          * -  
            - Single-threaded (non-threaded) |short_name|
            - Multi-threaded (internally threaded) |short_name| 
          * - Static linking
            - daal_core.lib, daal_sequential.lib
            - daal_core.lib, daal_thread.lib  
          * - Dynamic linking
            - daal_core_dll.lib 
            - aal_core_dll.lib 

Applications on Linux
---------------------

.. note::

  Known issues that you might encounter:

  - Static linking results in :ref:`incorrect linker behavior <issue_incorrect_linker_behavior>`
  - :ref:`No Level Zero in your environment <issue_level_zero>`

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

     .. list-table::
          :widths: 25 25 25
          :header-rows: 1
          :align: left

          * -  
            - Single-threaded (non-threaded) |short_name|
            - Multi-threaded (internally threaded) |short_name| 
          * - Static linking
            - libdaal_core.a, libdaal_sequential.a 
            - libdaal_core.a, libdaal_thread.a 
          * - Dynamic linking
            - libdaal_core.so, libdaal_sequential.so
            - libdaal_core.so, libdaal_thread.so

  - Add an additional |short_name| library:
   
    .. code-block:: text 
      
     -foffload-static-lib=<install dir>/daal/latest/libintel64/libdaal_sycl.a

Examples 
********

Dynamic linking, Multi-threaded |short_name|:

.. code-block:: text

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group -L<install dir>/daal/latest/lib/intel64 -ldaal_core -ldaal_thread.so -lpthread -ldl -lOpenCL -L<install dir>/tbb/latest/lib/intel64/gcc4.8 -ltbb -ltbbmalloc -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libdaal_sycl.a -Wl,--end-group

Static linking, Single-threaded |short_name|:

.. code-block:: text

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group <install dir>/daal/latest/lib/intel64/libdaal_core.a <install dir>/daal/latest/lib/intel64/libdaal_sequential.a -lpthread -ldl -lOpenCL -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libdaal_sycl.a -Wl,--end-group
