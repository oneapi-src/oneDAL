/* file: ResultId.java */
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
 * @ingroup zscore
 * @{
 */
package com.intel.daal.algorithms.normalization.zscore;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__RESULTID"></a>
 * \brief Available identifiers of results of the Z-score normalization algorithm
 */
public final class ResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int normalizedDataId = 0;
    private static final int meansId = 1;
    private static final int variancesId = 2;
    public static final ResultId normalizedData = new ResultId(normalizedDataId);
    public static final ResultId means = new ResultId(meansId);
    public static final ResultId variances = new ResultId(variancesId);
}
/** @} */
