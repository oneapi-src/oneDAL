/* file: SplitCriterionId.java */
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
