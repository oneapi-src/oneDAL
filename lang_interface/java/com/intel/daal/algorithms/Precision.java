/* file: Precision.java */
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
}
/** @} */
