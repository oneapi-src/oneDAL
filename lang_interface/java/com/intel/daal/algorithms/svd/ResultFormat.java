/* file: ResultFormat.java */
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__RESULTFORMAT"></a>
 * @brief Available options to return result matrices
 */
public final class ResultFormat {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result format object using the provided value
     * @param value     Value corresponding to the result format object
     */
    public ResultFormat(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result format object
     * @return Value corresponding to the result format object
     */
    public int getValue() {
        return _value;
    }

    private static final int notRequiredId          = 0;
    private static final int requiredInPackedFormId = 1;

    /** Matrix is not required */
    public static final ResultFormat notRequired          = new ResultFormat(notRequiredId);
    /** Matrix in the packed format is required */
    public static final ResultFormat requiredInPackedForm = new ResultFormat(requiredInPackedFormId);
}
/** @} */
