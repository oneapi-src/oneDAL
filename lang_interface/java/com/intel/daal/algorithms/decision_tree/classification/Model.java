/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @defgroup decision_tree_classification Classification
 * @brief Contains classes for decision tree classification algorithm
 * @ingroup decision_tree
 * @{
 */

package com.intel.daal.algorithms.decision_tree.classification;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__MODEL"></a>
 * @brief %Model of the classifier trained by decision tree classification algorithm in batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

}
/** @} */
