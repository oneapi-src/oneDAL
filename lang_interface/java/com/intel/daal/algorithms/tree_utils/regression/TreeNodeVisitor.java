/* file: TreeNodeVisitor.java */
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
 * @defgroup decision_tree_classification Classification
 * @brief Interface of callback object for classification model traversal
 * @ingroup decision_tree
 * @{
 */
/**
 * @brief Interface of callback object for classification model traversal
 */
package com.intel.daal.algorithms.tree_utils.regression;


import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS_REGRESSION__TREENODEVISITOR"></a>
 * @brief Interface of callback object for regression model traversal
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
