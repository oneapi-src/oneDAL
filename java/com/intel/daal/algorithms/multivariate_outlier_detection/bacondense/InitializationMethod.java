/* file: InitializationMethod.java */
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
 * @ingroup multivariate_outlier_detection_bacondense
 * @{
 */
package com.intel.daal.algorithms.multivariate_outlier_detection.bacondense;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__BACONDENSE__INITIALIZATIONMETHOD"></a>
 * \brief Available initialization methods for the BACON multivariate outlier detection algorithm \DAAL_DEPRECATED_USE{com.intel.daal.algorithms.bacon_outlier_detection.InitializationMethod}
 */
@Deprecated
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
