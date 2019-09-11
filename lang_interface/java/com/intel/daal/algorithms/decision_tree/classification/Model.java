/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @defgroup decision_tree_classification Classification
 * @brief Contains classes for decision tree classification algorithm
 * @ingroup decision_tree
 * @{
 */

package com.intel.daal.algorithms.decision_tree.classification;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__MODEL"></a>
 * @brief %Model of the classifier trained by decision tree classification algorithm in batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     *  Perform Depth First Traversal of a tree in the model
     * @DAAL_DEPRECATED
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseDF(com.intel.daal.algorithms.classifier.TreeNodeVisitor visitor) {
        cTraverseDF(this.cObject, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @DAAL_DEPRECATED
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBF(com.intel.daal.algorithms.classifier.TreeNodeVisitor visitor) {
        cTraverseBF(this.cObject, visitor);
    }

    /**
     *  Perform Depth First Traversal of a tree in the model
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseDFS(com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitor) {
        cTraverseDFS(this.cObject, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBFS(com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitor) {
        cTraverseBFS(this.cObject, visitor);
    }

    private native void cTraverseDF(long modAddr, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitorObj);
    private native void cTraverseBF(long modAddr, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitorObj);

    private native void cTraverseDFS(long modAddr, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitorObj);
    private native void cTraverseBFS(long modAddr, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitorObj);

}
/** @} */
