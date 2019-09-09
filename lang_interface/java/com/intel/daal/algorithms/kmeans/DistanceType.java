/* file: DistanceType.java */
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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTANCETYPE"></a>
 * @brief Available distance types for the K-Means algorithm
 */
public final class DistanceType {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the distance type object using the provided value
     * @param value     Value corresponding to the distance type object
     */
    public DistanceType(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the distance type object
     * @return Value corresponding to the distance type object
     */
    public int getValue() {
        return _value;
    }

    private static final int euclideanValue = 0;

    public static final DistanceType euclidean = new DistanceType(euclideanValue); /*!< Euclidean distance */
}
/** @} */
