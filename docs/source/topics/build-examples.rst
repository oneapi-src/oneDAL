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

.. |dpcpp_gsg| replace:: Get Started with Intel\ |reg|\  oneAPI DPC++/C++ Compiler
.. _dpcpp_gsg: https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html

Build and Run Examples
~~~~~~~~~~~~~~~~~~~~~~~

Perform the following steps to build and run examples demonstrating the
basic usage scenarios of |short_name| with SYCL*. Go to
:file:`<install_dir>/daal/<version>` and then set up an environment as shown in the example below:

.. note::

   All content below that starts with ``#`` is considered a comment and
   should not be run with the code.

1. Set up the required environment for |short_name|
   (variables such as ``CPATH``, ``LIBRARY_PATH``, and ``LD_LIBRARY_PATH``):

  .. tabs::

    .. group-tab:: Linux

      On Linux, there are two possible ways to set up the required environment:
      via ``vars.sh`` script or via ``modulefiles``.

      * Setting up |short_name| environment via ``vars.sh`` script

        Run the following command:

          .. prompt:: bash

            source ./env/vars.sh

      * Setting up |short_name| environment via ``modulefiles``

        1. Initialize ``modules``:

          .. prompt:: bash

            source $MODULESHOME/init/bash

          .. note:: Refer to `Environment Modules documentation <https://modules.readthedocs.io/en/latest/index.html>`_ for details.

        2. Provide ``modules`` with a path to the ``modulefiles`` directory:

          .. prompt:: bash

            module use ./modulefiles

        3. Run the module:

          .. prompt:: bash

            module load dal    

    .. group-tab:: Windows

      Run the following command:

      .. prompt:: bash

        /env/vars.bat

2. Copy ``./examples/daal/cpp_sycl`` to a writable directory if necessary (since it creates temporary files):

  .. prompt:: bash

    cp –r ./examples/daal/cpp_sycl ${WRITABLE_DIR}

3. Set up the compiler environment for |dpcpp|.
   See |dpcpp_gsg|_ for details.

4. Build and run DPC++ examples:

  .. note::

    You need to have write permissions to the :file:`examples` folder
    to build examples, and execute permissions to run them.
    Otherwise, you need to copy :file:`examples/daal/cpp_sycl` and :file:`examples/daal/data` folders
    to the directory with right permissions. These two folders must be retained
    in the same directory level relative to each other.

  .. tabs::

    .. group-tab:: Linux

      .. prompt:: bash

        # Navigate to DPC++ examples directory and build examples
        cd /examples/daal/cpp_sycl
        make sointel64 example=cor_dense_batch # This will compile and run Correlation example using Intel(R) oneAPI DPC++/C++ Compiler
        make sointel64 mode=build			   # This will compile all DPC++ examples

    .. group-tab:: Windows

      .. prompt:: bash

        # Navigate to DPC++ examples directory and build examples
        cd /examples/daal/cpp_sycl
        nmake libintel64 example=cor_dense_batch+ # This will compile and run Correlation example using Intel(R) oneAPI DPC++/C++ compiler
        nmake libintel64 mode=build			     # This will compile all DPC++ examples

  To see all avaliable parameters of the build procedure, type ``make`` on Linux\* or ``nmake`` on Windows\*.

5. The resulting example binaries and log files are written into the :file:`_results` directory.

  .. note::

    You should run DPC++ examples from :file:`examples/daal/cpp_sycl` folder, not from :file:`_results` folder.
    Most examples require data to be stored in :file:`examples/daal/data` folder and to have a relative link to it
    started from :file:`examples/daal/cpp_sycl` folder.


  You can build traditional C++ examples located in ``examples/daal/cpp`` folder in a similar way.

