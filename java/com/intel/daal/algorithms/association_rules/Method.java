/* file: Method.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__METHOD"></a>
 * @brief Available methods for computing large itemsets and association rules
 */
public final class Method {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    public Method(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    private static final int   Apriori = 0;
    public static final Method apriori = new Method(Apriori); /*!< Apriori method */
}
/** @} */
