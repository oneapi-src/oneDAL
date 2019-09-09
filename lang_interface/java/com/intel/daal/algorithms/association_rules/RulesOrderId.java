/* file: RulesOrderId.java */
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
 * @ingroup association_rules
 * @{
 */
package com.intel.daal.algorithms.association_rules;

import java.lang.annotation.Native;

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

    @Native private static final int RulesUnsorted           = 0;
    @Native private static final int RulesSortedByConfidence = 1;

    /** Unsorted */
    public static final RulesOrderId rulesUnsorted           = new RulesOrderId(RulesUnsorted);
    /** Sorted by confidence */
    public static final RulesOrderId rulesSortedByConfidence = new RulesOrderId(RulesSortedByConfidence);
}
/** @} */
