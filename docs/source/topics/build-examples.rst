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

.. |dpcpp_gsg| replace:: Get Started with Intel\ |reg|\  oneAPI DPC++ Compiler
.. _dpcpp_gsg: https://software.intel.com/en-us/get-started-with-dpcpp-compiler

Build and Run Examples
~~~~~~~~~~~~~~~~~~~~~~~

Perform the following steps to build and run examples demonstrating the
basic usage scenarios of |short_name| with SYCL*. Go to
:file:`<install_dir>/daal/<version>` and then set up an environment as shown in the example below:

.. note::

   All content starting with a # below is considered a comment and
   should not be run with the code.

1. Set up the |short_name| environment:

  **Linux\*:**

    .. substitution-prompt:: bash

      # Run script to setup CPATH, LIBRARY_PATH and LD_LIBRARY_PATH for |short_name|
      source ./env/vars.sh

  **Windows\*:**

    .. substitution-prompt:: bash

      # Run script to setup PATH, LIB and INCLUDE for |short_name|
      /env/vars.bat

2. Copy ``./examples/cpp_sycl`` to a writable directory if necessary:

  .. prompt:: bash

    # If necessary, copy ./examples/cpp_sycl to a writable directory (since it creates temporary files)
    cp –r ./examples/cpp_sycl ${WRITABLE_DIR}

3. Set up the compiler environment for |dpcpp|.
   See |dpcpp_gsg|_ for details.

4. Build and run DPC++ examples:

  .. note::

    You need to have write permissions to the :file:`examples` folder
    to build examples, and execute permissions to run them.
    Otherwise, you need to copy :file:`cpp_sycl` and :file:`data` folders
    to the directory with right permissions. These two folders must be retained
    in the same directory level relative to each other.

  **Linux\*:**

    .. prompt:: bash

      # Navigate to DPC++ examples directory and build examples
      cd /examples/cpp_sycl
      make sointel64 example=cor_dense_batch # This will compile and run Correlation example using Intel(R) oneAPI DPC++ Compiler
      make sointel64 mode=build			   # This will compile all DPC++ examples

  **Windows\*:**

    .. prompt:: bash

      # Navigate to DPC++ examples directory and build examples
      cd /examples/cpp_sycl
      nmake libintel64 example=cor_dense_batch+ # This will compile and run Correlation example using Intel(R) oneAPI DPC++ compiler
      nmake libintel64 mode=build			     # This will compile all DPC++ examples

  To see all avaliable parameters of the build procedure, type ``make`` on Linux\* or ``nmake`` on Windows\*.

5. The resulting example binaries and log files are written into the :file:`_results` directory.

  .. note::

    You should run DPC++ examples from :file:`cpp_sycl` folder, not from :file:`_results` folder.
    Most examples require data to be stored in :file:`examples\\data` folder and to have a relative link to it
    started from :file:`cpp_sycl` folder.


  You can build traditional C++ examples located in ``examples/cpp`` folder in a similar way.

