/* file: ResultId.java */
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

package com.intel.daal.algorithms.covariance;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__RESULTID"></a>
 * @brief Available result identifiers for the correlation or variance-covariance matrix algorithm
 */
public final class ResultId {
    private int _value;

    public ResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int covarianceValue = 0;
    private static final int meanValue       = 1;

    public static final ResultId covariance  = new ResultId(covarianceValue);  /*!< Variance-Covariance matrix */
    public static final ResultId correlation = new ResultId(covarianceValue);  /*!< Correlation matrix */
    public static final ResultId mean        = new ResultId(meanValue);        /*!< Vector of means */
}
