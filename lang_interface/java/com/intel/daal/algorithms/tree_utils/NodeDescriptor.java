/* file: NodeDescriptor.java */
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
 * @ingroup tree_utils
 * @{
 */
package com.intel.daal.algorithms.tree_utils;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__NODEDESCRIPTOR"></a>
 * @brief Struct containing base description of node in descision tree
 */
public class NodeDescriptor {
    public long level;
    public double impurity;
    public long nNodeSampleCount;
}
/** @} */
