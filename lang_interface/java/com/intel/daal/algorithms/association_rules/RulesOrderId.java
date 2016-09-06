/* file: RulesOrderId.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.association_rules;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RULESORDERID"></a>
 * @brief Available sort order options for resulting association rules
 */
public final class RulesOrderId {
    private int _value;

    public RulesOrderId(int value) {
        _value = value;
    }

    /**
     * Returns the value of the sort order option identifier
     * @return Value of the sort order option identifier
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
