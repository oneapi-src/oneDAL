/* file: ModelBuilder.java */
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
 * @defgroup gbt Gradient Boosted Trees
 * @brief Contains base classes of the gradient boosted trees algorithm
 * @ingroup training_and_prediction
 * @{
 */

package com.intel.daal.algorithms.gbt.classification;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__MODEL__BUILDER"></a>
 * @brief %Class for building model of the gradient boosted trees classification algorithm
 */
public class ModelBuilder extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /*!< %Reserved value for root nodes */
    public static final long noParent = -1;

    /**
     * Constructs the gradient boosted trees classification model builder
     * @param context       Context to manage gradient boosted trees classification model builder
     * @param nFeatures     Number of features in training data
     * @param nIterations   Number of trees in model for each class
     * @param nClasses      Number of classes
     */
    public ModelBuilder(DaalContext context, long nFeatures, long nIterations, long nClasses) {
        super(context);
        this.cObject = cInit(nFeatures, nIterations, nClasses);
    }

    /**
     * Create certain tree in the gradient boosted trees classification model for certain class
     * @param nNodes       Number of nodes in created tree
     * @param classLabel   Label of class for which tree is created. classLabel bellows  interval from 0 to (nClasses - 1)
     * @return             Positive number tree identifier
     */
    public long createTree(long nNodes, long classLabel) {
        return cCreateTree(this.cObject, nNodes, classLabel);
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
     * Get built model of gradient boosted trees classification
     * @return Model of gradient boosted trees classification
     */
    public Model getModel() {
        return new Model(getContext(), cGetModel(this.cObject));
    }


    private native long cInit(long nFeatures, long nIterations, long nClasses);
    private native long cCreateTree(long algAddr, long nNodes, long classLabel);
    private native long cAddSplitNode(long algAddr, long treeId, long parentId, long position, long featureIndex, double featureValue);
    private native long cAddLeafNode(long algAddr, long treeId, long parentId, long position, double response);
    private native long cGetModel(long algAddr);
}
/** @} */
