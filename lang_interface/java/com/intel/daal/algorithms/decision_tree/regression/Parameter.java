/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup decision_tree_regression
 */
/**
 * @brief Contains parameter for decision tree regression algorithm
 */
package com.intel.daal.algorithms.decision_tree.regression;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.PruningId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PARAMETER"></a>
 * @brief Parameter of the decision tree regression algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Returns the pruning method for decision tree training algorithm
     * @return Pruning method for decision tree
     */
    public PruningId getPruning() {
        return new PruningId(cGetPruning(this.cObject));
    }

    /**
     * Sets the pruning method for decision tree training algorithm
     * @param value   Pruning method for decision tree
     */
    public void setPruning(PruningId value) {
        cSetPruning(this.cObject, value.getValue());
    }

    /**
     * Returns the maximum tree depth. 0 means unlimited depth.
     * @return Maximum tree depth
     */
    public long getMaxTreeDepth() {
        return cGetMaxTreeDepth(this.cObject);
    }

    /**
     * Sets the maximum tree depth, 0 means unlimited depth
     * @param value   Maximum tree depth
     */
    public void setMaxTreeDepth(long value) {
        cSetMaxTreeDepth(this.cObject, value);
    }

    /**
     * Returns the minimum number of observations in the leaf node
     * @return Minimum number of observations in the leaf node
     */
    public long getMinObservationsInLeafNodes() {
        return cGetMinObservationsInLeafNodes(this.cObject);
    }

    /**
     * Sets the minimum number of observations in the leaf node
     * @param value   Minimum number of observations in the leaf node
     */
    public void setMinObservationsInLeafNodes(long value) {
        cSetMinObservationsInLeafNodes(this.cObject, value);
    }

    private native int  cGetPruning(long parAddr);
    private native void cSetPruning(long parAddr, int value);

    private native long cGetMaxTreeDepth(long parAddr);
    private native void cSetMaxTreeDepth(long parAddr, long value);

    private native long cGetMinObservationsInLeafNodes(long parAddr);
    private native void cSetMinObservationsInLeafNodes(long parAddr, long value);
}
/** @} */
