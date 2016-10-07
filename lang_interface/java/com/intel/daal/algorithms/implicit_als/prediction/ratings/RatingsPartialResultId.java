/* file: RatingsPartialResultId.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.implicit_als.prediction.ratings;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSPARTIALRESULTID"></a>
 * @brief Available identifiers of input partial model objects for the rating prediction stage
 *        of the implicit ALS algorithm
 */
public final class RatingsPartialResultId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value of the input object identifier
     */
    public RatingsPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value of the input object identifier
     * @return Value of the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int finalResultId  = 0;

    /** Result of the implicit ALS ratings prediction algorithm */
    public static final RatingsPartialResultId finalResult  = new RatingsPartialResultId(finalResultId);
}
