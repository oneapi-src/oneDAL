/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup decision_tree_classification
 */
/**
 * @brief Contains classes of the decision tree classification algorithm
 */
package com.intel.daal.algorithms.decision_tree.classification;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.PruningId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__TRAINING__PARAMETER"></a>
 * @brief Base class for parameters of the decision tree classification algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

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

    /**
     * Returns the split criterion for decision tree classification
     * @return Split criterion for decision tree classification
     */
    public SplitCriterionId getSplitCriterion() {
        return new SplitCriterionId(cGetSplitCriterion(this.cObject));
    }

    /**
     * Sets the split criterion for decision tree classification
     * @param value   Split criterion for decision tree classification
     */
    public void setSplitCriterion(SplitCriterionId value) {
        cSetSplitCriterion(this.cObject, value.getValue());
    }

    private native int  cGetPruning(long parAddr);
    private native void cSetPruning(long parAddr, int value);

    private native long cGetMaxTreeDepth(long parAddr);
    private native void cSetMaxTreeDepth(long parAddr, long value);

    private native long cGetMinObservationsInLeafNodes(long parAddr);
    private native void cSetMinObservationsInLeafNodes(long parAddr, long value);

    private native int  cGetSplitCriterion(long parAddr);
    private native void cSetSplitCriterion(long parAddr, int value);

}
/** @} */
