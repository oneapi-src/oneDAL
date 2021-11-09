.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

The DBSCAN algorithm solves :capterm:`clustering` problem by the following way:
given a set of observations, it groups together ones that are closely packed
(having at least :math:`min_observations` number of neighbors inside distance :math:`epsilon`)
and marking other ones as outliers.

.. |c_math| replace:: :ref:`Compute <dbscan_c_math>`
.. |c_bf| replace:: :ref:`Brute Force <dbscan_c_brute_force>`
.. |c_input| replace:: :ref:`compute_input <dbscan_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <dbscan_c_api_result>`
.. |c_op| replace:: :ref:`compute(...) <dbscan_c_api>`

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |c_math|             |c_bf|               |c_op|   |c_input|   |c_result|
=============== =========================== ======== =========== ============

