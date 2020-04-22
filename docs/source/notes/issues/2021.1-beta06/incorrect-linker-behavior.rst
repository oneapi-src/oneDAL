.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. |dpcpp_comp| replace:: Intel\ |reg|\  oneAPI DPC++ Compiler
.. _dpcpp_comp: https://software.intel.com/en-us/oneapi/dpc-compiler

.. _issue_incorrect_linker_behavior:

Incorrect linker behavior
*************************

When you build oneDAL application using clang++ and static linking, you might encounter the following issue:

.. code-block:: text

  /usr/bin/ld: BFD (GNU Binutils for Ubuntu) <version> assertion fail ../../bfd/elf.c

|dpcpp_comp|_ produces zero size relocation section in |short_name| libraries, which causes incorrect linker behavior.

How to Fix
----------

Use linker keys ``-Bstatic`` and ``-Bdynamic`` for linking |short_name| libs. For example:

- Previous linking approach:

  ::
    
    -Wl, --start-group 
    <daal_lib_path>/libdaal_core.a <daal_lib_path>/libdaal_thread.a

- New linking approach:

  .. code-block::
    :emphasize-lines: 2,4
    
    -Wl, --start-group 
    -Wl, -L<daal_lib_path>, -Bstatic,
    -ldaal_core, -ldaal_thread, 
    -Bdynamic
