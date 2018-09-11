/* file: RatingsPartialModelInputId.java */
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
 * @ingroup implicit_als_prediction
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSPARTIALMODELINPUTID"></a>
 * @brief Available identifiers of input PartialModel objects for the rating prediction stage
 *        of the implicit ALS algorithm
 */
public final class RatingsPartialModelInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the ratings input partial model object identifier using the provided value
     * @param value     Value corresponding to the ratings input partial model object identifier
     */
    public RatingsPartialModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the ratings input partial model object identifier
     * @return Value corresponding to the ratings input partial model object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int usersPartialModelId = 0;
    private static final int itemsPartialModelId = 1;

    /** %Input partial model containing users factors trained by the implicit ALS algorithm in the distributed processing mode */
    public static final RatingsPartialModelInputId usersPartialModel = new RatingsPartialModelInputId(usersPartialModelId);
    /** %Input partial model containing items factors trained by the implicit ALS algorithm in the distributed processing mode */
    public static final RatingsPartialModelInputId itemsPartialModel = new RatingsPartialModelInputId(itemsPartialModelId);
}
/** @} */
