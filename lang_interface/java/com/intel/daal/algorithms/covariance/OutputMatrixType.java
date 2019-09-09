/* file: OutputMatrixType.java */
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__OUTPUTMATRIXTYPE"></a>
 * @brief Available types of the computed correlation or variance-covariance matrix
 */
public final class OutputMatrixType {
    private int _value;

    /**
     * Constructs the computed correlation or variance-covariance matrix object using the provided value
     * @param value     Value corresponding to the computed correlation or variance-covariance matrix object
     */
    public OutputMatrixType(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the computed correlation or variance-covariance matrix object
     * @return Value corresponding to the computed correlation or variance-covariance matrix object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int CovarianceMatrix  = 0;
    @Native private static final int CorrelationMatrix = 1;

    public static final OutputMatrixType covarianceMatrix  = new OutputMatrixType(
            CovarianceMatrix);  /*!< Variance-Covariance matrix */
    public static final OutputMatrixType correlationMatrix = new OutputMatrixType(
            CorrelationMatrix);  /*!< Correlation matrix */
}
/** @} */
