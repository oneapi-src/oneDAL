/* file: SplitCriterionId.java */
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
 * @ingroup decision_tree_classification
 * @{
 */
package com.intel.daal.algorithms.decision_tree.classification;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__SPLIT_CRITERIONID"></a>
 * @brief Split criterion for Decision tree classification algorithm
 */
public final class SplitCriterionId {
    private int _value;

    /**
     * Constructs the split criterion object identifier using the provided value
     * @param value     Value corresponding to the split criterion object identifier
     */
    public SplitCriterionId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the split criterion object identifier
     * @return Value corresponding to the split criterion object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int giniId     = 0;
    private static final int infoGainId = 1;

    public static final SplitCriterionId gini     = new SplitCriterionId(giniId);     /*!< Gini index */
    public static final SplitCriterionId infoGain = new SplitCriterionId(infoGainId); /*!< Information gain */
}
/** @} */
