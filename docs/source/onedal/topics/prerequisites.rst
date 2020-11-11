.. ******************************************************************************
.. * Copyright 2019-2020 Intel Corporation
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

.. |dpcpp_comp| replace:: Intel\ |reg|\  oneAPI DPC++/C++ Compiler
.. _dpcpp_comp: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html

.. _before_you_begin:

Before You Begin
~~~~~~~~~~~~~~~~

|short_name| is located in :file:`<install_dir>/daal` directory where :file:`<install_dir>`
is the directory in which Intel\ |reg|\  oneAPI Toolkit was installed.

The current version of |short_name| with
DPC++ is available for Linux\* and Windows\* 64-bit operating systems. The
prebuilt |short_name| libraries can be found in the :file:`<install_dir>/daal/<version>/redist`
directory.

The dependencies needed to build examples with DPC++ extensions API are:

- C/C++ Compiler with C++11 support (or C++14 support on Windows\*)
- |dpcpp_comp|_ 2019 August release or later (for DPC++ support)
- OpenCL™ runtime 1.2 or later (to run the DPC++ runtime)
- GNU\* Make on Linux\*, nmake on Windows\*

