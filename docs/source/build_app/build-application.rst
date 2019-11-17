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

Build application with oneDAL
=============================

See Get Started Guides for `Windows\* <https://software.intel.com/en-us/get-started-with-daal-for-windows>`_ 
and `Linux\* <https://software.intel.com/en-us/get-started-with-daal-for-linux>`_ 
for instruction on how to build applications for C++.
This section contains instructions for building applications with oneDAL for SYCL\*.

Applications on Windows
-----------------------

#. Download and install |base_tk|.

#. In Microsoft Visual Studio* Integrated Development environment (IDE), 
   open or create a C++ project for your |short_name| application to build.

#. In project properties set "Intel(R) oneAPI DPC++ Compiler" platform toolset:

   .. image:: ./images/MSVSPlatformToolset.jpg
     :width: 600
     :align: center

#. In project properties add oneDAL includes folder to Additional Include Directories:


   .. image:: ./images/AdditionalIncludeDirs.jpg
     :width: 600
     :align: center
    

#. In project properties add oneDAL and TBB libraries folders to Library Directories:

   .. image:: ./images/LibraryDirectories.jpg
     :width: 600
     :align: center

#. In project properties add oneDAL and OpenCL libraries to Additional Dependencies:

   .. image:: ./images/AdditionalDependencies.jpg
     :width: 600
     :align: center

#. Add the appropriate libraries to your project based on oneDAL threading mode and linking method:

     .. list-table::
          :widths: 25 25 25
          :header-rows: 1
          :align: left

          * -  
            - Single-threaded (non-threaded) oneDAL
            - Multi-threaded (internally threaded) oneDAL 
          * - Static linking
            - daal_core.lib, daal_sequential.lib
            - daal_core.lib, daal_thread.lib  
          * - Dynamic linking
            - daal_core_dll.lib 
            - aal_core_dll.lib 

Applications on Linux
---------------------

#. Download and install |base_tk|.

#. Set environment variables by calling ``<install dir>/setvars.sh``.

#. Build your application with clang++:

   - Add ``fsycl`` option to the command: 
   
     :: 
     
       -fsycl

   - Add ``ONEAPI_DAAL_USE_MKL_GPU_GEMM`` definition:
   
     :: 
     
        -DONEAPI_DAAL_USE_MKL_GPU_GEMM

   - Add oneDAL includes folder:
   
     :: 
     
        -I<install dir>/daal/latest/include

   - Add oneDAL libraries. Choose the appropriate oneDAL libraries based on oneDAL threading mode and linking method:

     .. list-table::
          :widths: 25 25 25
          :header-rows: 1
          :align: left

          * -  
            - Single-threaded (non-threaded) oneDAL
            - Multi-threaded (internally threaded) oneDAL 
          * - Static linking
            - libdaal_core.a, libdaal_sequential.a 
            - libdaal_core.a, libdaal_thread.a 
          * - Dynamic linking
            - libdaal_core.so, libdaal_sequential.so
            - libdaal_core.so, libdaal_thread.so

  - Add an additional oneDAL library:
   
    :: 
      
     -foffload-static-lib=<install dir>/daal/latest/libintel64/libdaal_sycl.a

Examples 
********

Dynamic linking, Multi-threaded oneDAL:

::

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group -L<install dir>/daal/latest/lib/intel64 -ldaal_core -ldaal_thread.so -lpthread -ldl -lOpenCL -L<install dir>/tbb/latest/lib/intel64/gcc4.8 -ltbb -ltbbmalloc -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libdaal_sycl.a -Wl,--end-group

Static linking, Single-threaded oneDAL:

::

     clang++ -fsycl -DONEAPI_DAAL_USE_MKL_GPU_GEMM my_first_daal_program.cpp -Wl,
     --start-group <install dir>/daal/latest/lib/intel64/libdaal_core.a <install dir>/daal/latest/lib/intel64/libdaal_sequential.a -lpthread -ldl -lOpenCL -foffload-static-lib=<install dir>/daal/latest/lib/intel64/libdaal_sycl.a -Wl,--end-group
