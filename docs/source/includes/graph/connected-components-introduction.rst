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

Connected components algorithm receives an undirected graph :math:`G` as input and searches connected components in :math:`G`.
The algorithm returns labels for the graph vertices such that vertices inside the same component have the same labels.

.. |cc_compute|     replace::   :ref:`Computing                   <connected_components_compute>`
.. |cc_afforest|    replace::   :ref:`afforest                    <connected_components_afforest>`
.. |cc_api|         replace::   :ref:`vertex_partitioning(...)    <connected_components_t_api>`
.. |cc_api_input|   replace::   :ref:`vertex_partitioning_input   <connected_components_t_api_input>`
.. |cc_api_result|  replace::   :ref:`vertex_partitioning_result  <connected_components_t_api_result>`

================ =========================== ============ ================= =================
 **Operation**     **Computational methods**           **Programming Interface**
---------------- --------------------------- ------------------------------------------------
  |cc_compute|         |cc_afforest|           |cc_api|    |cc_api_input|    |cc_api_result|
================ =========================== ============ ================= =================
