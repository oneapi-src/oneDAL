/* file: Method.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__METHOD"></a>
 * @brief Available methods of the K-Means algorithm
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

    private static final int lloydDenseValue = 0;
    private static final int lloydCSRValue   = 1;

    public static final Method defaultDense = new Method(lloydDenseValue); /*!< Default: performance-oriented method, synonym of lloydDense */
    public static final Method lloydDense   = new Method(lloydDenseValue); /*!< Default: performance-oriented method, synonym of defaultDense */
    public static final Method lloydCSR     = new Method(lloydCSRValue);   /*!< Method for sparse data in the CSR format */
}
/** @} */
