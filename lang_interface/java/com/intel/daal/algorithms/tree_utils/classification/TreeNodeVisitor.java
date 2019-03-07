/* file: TreeNodeVisitor.java */
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
 * @brief Interface of callback object for classification model traversal
 * @ingroup decision_tree
 * @{
 */
/**
 * @brief Interface of callback object for classification model traversal
 */
package com.intel.daal.algorithms.tree_utils.classification;

import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS_CLASSIFICATION__TREENODEVISITOR"></a>
 * @brief Interface of callback object for classification model traversal
 */
public abstract class TreeNodeVisitor {

    /**
     * Default constructor
     */
    public TreeNodeVisitor() {}

    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  @param desc Strucutre describing the split node of the tree
    *  @return This method should return false to cancel further search and true otherwise
    */
    public abstract boolean onLeafNode(LeafNodeDescriptor desc);
    /**
    *  This method is called by traversal method when a split node is visited.
    *  @param desc Strucutre describing the split node of the tree
    *  @return This method should return false to cancel further search and true otherwise
    */
    public abstract boolean onSplitNode(SplitNodeDescriptor desc);
}
/** @} */
