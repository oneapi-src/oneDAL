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



Decision Forest
---------------

The library provides decision forest classification and regression
algorithms based on an ensemble of tree-structured classifiers
(decision trees) built using the general technique of bootstrap
aggregation (bagging) and random choice of features. *Decision tree*
is a binary tree graph. Its internal (split) nodes represent a
*decision function* used to select the following (child) node at the
prediction stage. Its leaf (terminal) nodes represent the
corresponding response values, which are the result of the prediction
from the tree. For more details, see Classification and Regression >
Decision Tree, [Breiman84]_ and [Breiman2001]_.

.. toctree::
   :maxdepth: 1
   
   decision-forest.rst
   decision-forest-regression
   decision-forest-classification
