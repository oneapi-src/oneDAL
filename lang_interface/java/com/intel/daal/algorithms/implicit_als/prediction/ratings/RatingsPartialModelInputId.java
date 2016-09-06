/* file: RatingsPartialModelInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSPARTIALMODELINPUTID"></a>
 * @brief Available identifiers of input PartialModel objects for the rating prediction stage
 *        of the implicit ALS algorithm
 */
public final class RatingsPartialModelInputId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value of the input object identifier
     */
    public RatingsPartialModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value of the input object identifier
     * @return Value of the input object identifier
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
