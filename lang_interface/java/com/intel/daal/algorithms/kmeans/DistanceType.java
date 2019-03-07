/* file: DistanceType.java */
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
