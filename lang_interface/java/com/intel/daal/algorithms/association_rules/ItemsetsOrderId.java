/* file: ItemsetsOrderId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__ITEMSETSORDERID"></a>
 * @brief Available sort order options for resulting itemsets
 */
public final class ItemsetsOrderId {
    private int _value;

    public ItemsetsOrderId(int value) {
        _value = value;
    }

    /**
     * Returns the value of the sort order option identifier
     * @return Value of the sort order option identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ItemsetsUnsorted        = 0;
    private static final int ItemsetsSortedBySupport = 1;

    /** Unsorted */
    public static final ItemsetsOrderId itemsetsUnsorted        = new ItemsetsOrderId(ItemsetsUnsorted);
    /** Sorted by the support value */
    public static final ItemsetsOrderId itemsetsSortedBySupport = new ItemsetsOrderId(ItemsetsSortedBySupport);
}
