/* file: RatingsPartialResultId.java */
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
 * @ingroup implicit_als_prediction
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSPARTIALRESULTID"></a>
 * @brief Available identifiers of input partial model objects for the rating prediction stage
 *        of the implicit ALS algorithm
 */
public final class RatingsPartialResultId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the ratings partial result object identifier using the provided value
     * @param value     Value corresponding to the ratings partial result object identifier
     */
    public RatingsPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the ratings partial result object identifier
     * @return Value corresponding to the ratings partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int finalResultId  = 0;

    /** Result of the implicit ALS ratings prediction algorithm */
    public static final RatingsPartialResultId finalResult  = new RatingsPartialResultId(finalResultId);
}
/** @} */
