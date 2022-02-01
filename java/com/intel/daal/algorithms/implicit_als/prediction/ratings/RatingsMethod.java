/* file: RatingsMethod.java */
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
