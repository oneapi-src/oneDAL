/* file: Model.java */
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
 * @defgroup decision_forest_classification Classification
 * @brief Contains classes for decision forest classification algorithm
 * @ingroup decision_forest
 * @{
 */

package com.intel.daal.algorithms.decision_forest.classification;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__MODEL"></a>
 * @brief %Model of the classifier trained by decision forest classification algorithm in batch processing mode.
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
     * Get number of trees in the model
     * @return number of trees
     */
    public long getNumberOfTrees() {
        return cGetNumberOfTrees(this.cObject);
    }

    /**
     *  Perform Depth First Traversal of a tree in the model
     * @DAAL_DEPRECATED
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseDF(long iTree, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitor) {
        cTraverseDF(this.cObject, iTree, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @DAAL_DEPRECATED
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBF(long iTree, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitor) {
        cTraverseBF(this.cObject, iTree, visitor);
    }

    /**
     *  Removes all trees from the model
     */
    public void clear() {
        cClear(this.cObject);
    }

    /**
     *  Perform Depth First Traversal of a tree in the model
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseDFS(long iTree, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitor) {
        cTraverseDFS(this.cObject, iTree, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBFS(long iTree, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitor) {
        cTraverseBFS(this.cObject, iTree, visitor);
    }

    private native long cGetNumberOfTrees(long modAddr);
    private native void cTraverseDF(long modAddr, long iTree, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitorObj);
    private native void cTraverseBF(long modAddr, long iTree, com.intel.daal.algorithms.classifier.TreeNodeVisitor visitorObj);
    private native void cClear(long modAddr);
    private native void cTraverseDFS(long modAddr, long iTree, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitorObj);
    private native void cTraverseBFS(long modAddr, long iTree, com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor visitorObj);
}
/** @} */
