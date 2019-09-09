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
 * @defgroup gbt_regression Regression
 * @brief Contains classes for gradient boosted trees regression algorithm
 * @ingroup gbt
 * @{
 */
/**
 * @brief Contains classes of the gradient boosted trees regression algorithm
 */
package com.intel.daal.algorithms.gbt.regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__MODEL"></a>
 * @brief %Model trained by gradient boosted trees regression algorithm in batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.Model {
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
    public void traverseDF(long iTree, com.intel.daal.algorithms.regression.TreeNodeVisitor visitor) {
        cTraverseDF(this.cObject, iTree, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @DAAL_DEPRECATED
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBF(long iTree, com.intel.daal.algorithms.regression.TreeNodeVisitor visitor) {
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
    public void traverseDFS(long iTree, com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor visitor) {
        cTraverseDFS(this.cObject, iTree, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @param iTree   Index of the tree to traverse
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBFS(long iTree, com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor visitor) {
        cTraverseBFS(this.cObject, iTree, visitor);
    }

    private native long cGetNumberOfTrees(long modAddr);
    private native void cTraverseDF(long modAddr, long iTree, com.intel.daal.algorithms.regression.TreeNodeVisitor visitorObj);
    private native void cTraverseBF(long modAddr, long iTree, com.intel.daal.algorithms.regression.TreeNodeVisitor visitorObj);
    private native void cClear(long modAddr);
    private native void cTraverseDFS(long modAddr, long iTree, com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor visitorObj);
    private native void cTraverseBFS(long modAddr, long iTree, com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor visitorObj);
}
/** @} */
