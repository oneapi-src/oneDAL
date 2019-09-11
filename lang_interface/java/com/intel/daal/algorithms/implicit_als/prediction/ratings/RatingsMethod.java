/* file: RatingsMethod.java */
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
 * @defgroup implicit_als_prediction Prediction
 * @brief Contains classes for making implicit ALS model-based prediction
 * @ingroup implicit_als
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
import java.io.Serializable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSMETHOD"></a>
 * @brief Available methods for computing the results of implicit ALS model-based ratings prediction
 */
public final class RatingsMethod implements Serializable {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;


    /**
     * Constructs the ratings method object using the provided value
     * @param value     Value corresponding to the ratings method object
     */
    public RatingsMethod(int value) {
        _value = value;
    }


    /**
     * Returns the value corresponding to the ratings method object
     * @return Value corresponding to the ratings method object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseId = 0;
    private static final int allUsersAllItemsId = 0;

    /** Default: predicts ratings based on the implicit ALS model and input data in the dense format */
    public static final RatingsMethod defaultDense = new RatingsMethod(defaultDenseId);

    /** Predicts ratings for all users and items based on the implicit ALS model and input data in the dense format */
    public static final RatingsMethod allUsersAllItems = new RatingsMethod(allUsersAllItemsId);
}
/** @} */
