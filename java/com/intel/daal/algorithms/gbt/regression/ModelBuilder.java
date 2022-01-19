/* file: ModelBuilder.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @defgroup gbt Gradient Boosted Trees
 * @brief Contains base classes of the gradient boosted trees algorithm
 * @ingroup training_and_prediction
 * @{
 */

package com.intel.daal.algorithms.gbt.regression;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__MODEL__BUILDER"></a>
 * @brief %Class for building model of the gradient boosted trees regression algorithm
 */
public class ModelBuilder extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /*!< %Reserved value for root nodes */
    public static final long noParent = -1;

    /**
     * Constructs the gradient boosted trees regression model builder
     * @param context       Context to manage gradient boosted trees regression model builder
     * @param nFeatures     Number of features in training data
     * @param nIterations   Number of trees in model and iterations performed on training stage of gbt algorithm
     */
    public ModelBuilder(DaalContext context, long nFeatures, long nIterations) {
        super(context);
        this.cObject = cInit(nFeatures, nIterations);
    }

    /**
     * Create certain tree in the gradient boosted trees regression model
     * @param nNodes       Number of nodes in created tree
     * @return             Positive number tree identifier
     */
    public long createTree(long nNodes) {
        return cCreateTree(this.cObject, nNodes);
    }

    /**
     * Create Split node and add it to certain tree
     * @param treeId        Tree to which new node is added
     * @param parentId      Parent node to which new node is added (use noParent for root node)
     * @param position      Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
     * @param featureIndex  Feature index for spliting
     * @param featureValue  Feature value for spliting
     * @return              Positive number node identifier
     */
    public long addSplitNode(long treeId, long parentId, long position, long featureIndex, double featureValue) {
        return cAddSplitNode(this.cObject, treeId, parentId, position, featureIndex, featureValue);
    }

    /**
     * Create Leaf node and add it to certain tree
     * @param treeId        Tree to which new node is added
     * @param parentId      Parent node to which new node is added (use noParent for root node)
     * @param position      Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
     * @param response      Response value for leaf node to be predicted
     * @return              Positive number node identifier
     */
    public long addLeafNode(long treeId, long parentId, long position, double response) {
        return cAddLeafNode(this.cObject, treeId, parentId, position, response);
    }

    /**
     * Get built model of gradient boosted trees regression
     * @return Model of gradient boosted trees regression
     */
    public Model getModel() {
        return new Model(getContext(), cGetModel(this.cObject));
    }


    private native long cInit(long nFeatures, long nIterations);
    private native long cCreateTree(long algAddr, long nNodes);
    private native long cAddSplitNode(long algAddr, long treeId, long parentId, long position, long featureIndex, double featureValue);
    private native long cAddLeafNode(long algAddr, long treeId, long parentId, long position, double response);
    private native long cGetModel(long algAddr);
}
/** @} */
