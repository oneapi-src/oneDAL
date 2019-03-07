/* file: PartialResultId.java */
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the SVD algorithm obtained in the online processing mode and in the first step in
 * the distributed processing mode
 */
public final class PartialResultId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public PartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep1ForStep3Id = 0;
    private static final int outputOfStep1ForStep2Id = 1;

    /** DataCollection with data computed in the first step to be transferred to the third step in the distributed processing mode */
    public static final PartialResultId outputOfStep1ForStep3 = new PartialResultId(outputOfStep1ForStep3Id);
    /** DataCollection with data computed in the first step to be transferred to the second step in the distributed processing mode */
    public static final PartialResultId outputOfStep1ForStep2 = new PartialResultId(outputOfStep1ForStep2Id);
}
/** @} */
