/* file: Precision.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import java.io.Serializable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PRECISION"></a>
 * @brief Available precisions for algorithms
 */
public final class Precision implements Serializable {
    private int _value;

    /**
     * Constructs the precision object using the provided value
     * @param value     Value corresponding to the precision object
     */
    public Precision(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the precision object
     * @return Value corresponding to the precision object
     */
    public int getValue() {
        return _value;
    }

    private static final int doublePrecisionValue = 0;
    private static final int singlePrecisionValue = 1;

    public static final Precision doublePrecision = new Precision(doublePrecisionValue); /* Double precision */
    public static final Precision singlePrecision = new Precision(singlePrecisionValue); /* Single precision */

    public static Precision fromClass(Class<? extends Number> cls) {
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("Unsupported precision type");
        }
        return (cls == Double.class) ? Precision.doublePrecision : Precision.singlePrecision;
    }
}
/** @} */
