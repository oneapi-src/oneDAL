/* file: InitializationMethod.java */
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
 * @ingroup bacon_outlier_detection_defaultdense
 * @{
 */
package com.intel.daal.algorithms.bacon_outlier_detection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__DEFAULTDENSE__INITIALIZATIONMETHOD"></a>
 * \brief Available initialization methods for the BACON multivariate outlier detection algorithm
 */
public final class InitializationMethod {
    private int _value;

    /**
     * Constructs the initialization method object using the provided value
     * @param value     Value corresponding to the initialization method object
     */
    public InitializationMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization method object
     * @return Value corresponding to the initialization method object
     */
    public int getValue() {
        return _value;
    }

    private static final int baconMedianValue      = 0;
    private static final int baconMahalanobisValue = 1;

    /** Median-based method */
    public static final InitializationMethod baconMedian = new InitializationMethod(baconMedianValue);

    /** Mahalanobis distance-based method */
    public static final InitializationMethod baconMahalanobis = new InitializationMethod(baconMahalanobisValue);
}
/** @} */
