/* file: RulesOrderId.java */
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
 * @ingroup association_rules
 * @{
 */
package com.intel.daal.algorithms.association_rules;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RULESORDERID"></a>
 * @brief Available sort order options for resulting association rules
 */
public final class RulesOrderId {
    private int _value;

    /**
     * Constructs the sort order object identifier using the provided value
     * @param value     Value corresponding to the sort order object identifier
     */
    public RulesOrderId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the sort order object identifier
     * @return Value corresponding to the sort order object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int RulesUnsorted           = 0;
    private static final int RulesSortedByConfidence = 1;

    /** Unsorted */
    public static final RulesOrderId rulesUnsorted           = new RulesOrderId(RulesUnsorted);
    /** Sorted by confidence */
    public static final RulesOrderId rulesSortedByConfidence = new RulesOrderId(RulesSortedByConfidence);
}
/** @} */
