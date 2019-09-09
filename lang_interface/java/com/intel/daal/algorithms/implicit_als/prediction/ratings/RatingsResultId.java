/* file: RatingsResultId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup implicit_als_prediction
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSRESULTID"></a>
 * @brief Available identifiers of the results of the rating prediction stage of the implicit ALS algorithm
 */
public final class RatingsResultId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the rating prediction stage result object identifier using the provided value
     * @param value     Value corresponding to the rating prediction stage result object identifier
     */
    public RatingsResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the rating prediction stage result object identifier
     * @return Value corresponding to the rating prediction stage result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int predictionId  = 0;

    /** Numeric table containing predicted ratings */
    public static final RatingsResultId prediction  = new RatingsResultId(predictionId);
}
/** @} */
