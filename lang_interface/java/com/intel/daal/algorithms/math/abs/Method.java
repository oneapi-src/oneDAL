/* file: Method.java */
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
 * @ingroup abs
 * @{
 */
package com.intel.daal.algorithms.math.abs;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__ABS__METHOD"></a>
 * @brief Available methods for absolute value function
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

    private static final int   DefaultMethodValue = 0;
    private static final int   FastCSRMethodValue = 1;
    public static final Method defaultDense       = new Method(DefaultMethodValue); /*!< Default method */
    public static final Method fastCSR            = new Method(FastCSRMethodValue); /*!< Default: performance-oriented method.
                                                                                    Works with Compressed Sparse Rows(CSR)
                                                                                    input numeric tables */
}
/** @} */
