/* file: SplitNodeDescriptor.java */
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
