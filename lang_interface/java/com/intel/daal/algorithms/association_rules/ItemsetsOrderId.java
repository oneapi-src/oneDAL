/* file: ItemsetsOrderId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__ITEMSETSORDERID"></a>
 * @brief Available sort order options for resulting itemsets
 */
public final class ItemsetsOrderId {
    private int _value;

    /**
     * Constructs the itemsets object identifier using the provided value
     * @param value     Value corresponding to the itemsets object identifier
     */
    public ItemsetsOrderId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the itemsets object identifier
     * @return Value corresponding to the itemsets object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int ItemsetsUnsorted        = 0;
    @Native private static final int ItemsetsSortedBySupport = 1;

    /** Unsorted */
    public static final ItemsetsOrderId itemsetsUnsorted        = new ItemsetsOrderId(ItemsetsUnsorted);
    /** Sorted by the support value */
    public static final ItemsetsOrderId itemsetsSortedBySupport = new ItemsetsOrderId(ItemsetsSortedBySupport);
}
/** @} */
