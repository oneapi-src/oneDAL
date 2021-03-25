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

.. default-domain:: cpp

.. _api_df:

==================================================
Decision Forest Classification and Regression (DF)
==================================================

.. include:: ../../../includes/ensembles/df-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Decision Forest Classification and Regression <alg_df>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::decision_forest`` namespace and are available via inclusion of the
``oneapi/dal/algo/decision_forest.hpp`` header file.
                   
Enum classes
------------
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::error_metric_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::variable_importance_mode 
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::infer_mode
.. onedal_enumclass:: oneapi::dal::decision_forest::v1::voting_mode
                   
Descriptor
++++++++++
.. onedal_class:: oneapi::dal::decision_forest::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::task::v1

Model
-----
.. onedal_class:: oneapi::dal::decision_forest::v1::model

.. _df_t_api:

Training :expr:`train(...)`
+++++++++++++++++++++++++++
.. _df_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::train_input


.. _df_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::train_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::train_result train(const Descriptor& desc, \
                                         const decision_forest::train_input& input)

   :param desc: Decision Forest algorithm descriptor :expr:`decision_forest::descriptor`.
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.is_empty == false`
      | :expr:`input.labels.is_empty == false`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.data.row_count == input.labels.row_count`
      | :expr:`desc.get_bootstrap() == true || 
               (desc.get_bootstrap() == false &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_raw &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_scaled)`
      | :expr:`desc.get_bootstrap() == true || 
               (desc.get_bootstrap() == false && desc.get_error_metric_mode() == error_metric_mode::none)`

.. _df_i_api:

Inference :expr:`infer(...)`
++++++++++++++++++++++++++++
.. _df_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::infer_input


.. _df_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::v1::infer_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::infer_result infer(const Descriptor& desc, \
                                         const decision_forest::infer_input& input)

   :param desc: Decision Forest algorithm descriptor :expr:`decision_forest::descriptor`.
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.is_empty == false`