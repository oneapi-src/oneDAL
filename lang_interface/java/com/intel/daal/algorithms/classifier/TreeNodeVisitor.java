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
package com.intel.daal.algorithms.classifier;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TREENODEVISITOR"></a>
 * @DAAL_DEPRECATED
 * @brief Interface of callback object for classification model traversal
 */
public abstract class TreeNodeVisitor {

    /**
     * Default constructor
     */
    public TreeNodeVisitor() {}

    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  @param level    Level in the tree where this node is located
    *  @param response The value of response given by that node
    *  @return This method should return false to cancel further search and true otherwise
    */
    public abstract boolean onLeafNode(long level, long response);
    /**
    *  This method is called by traversal method when a split node is visited.
    *  @param level        Index of the feature used in a split criteria
    *  @param featureIndex Index of the feature used as a split criteria in this node
    *  @param featureValue Feature value used as a split criteria in this node
    *  @return This method should return false to cancel further search and true otherwise
    */
    public abstract boolean onSplitNode(long level, long featureIndex, double featureValue);
}
/** @} */
