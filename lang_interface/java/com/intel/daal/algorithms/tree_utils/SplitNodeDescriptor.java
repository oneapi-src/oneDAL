/* file: SplitNodeDescriptor.java */
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
 * @ingroup tree_utils
 * @{
 */
package com.intel.daal.algorithms.tree_utils;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__SPLITNODEDESCRIPTOR"></a>
 * @brief Struct containing description of split node in descision tree
 */
public final class SplitNodeDescriptor extends NodeDescriptor {
    public long featureIndex;
    public double featureValue;

    public SplitNodeDescriptor(long level_, long featureIndex_, double featureValue_, double impurity_, long nNodeSampleCount_)
    {
        super.level = level_;
        featureIndex = featureIndex_;
        featureValue = featureValue_;
        super.impurity = impurity_;
        super.nNodeSampleCount = nNodeSampleCount_;
    }
}
/** @} */
