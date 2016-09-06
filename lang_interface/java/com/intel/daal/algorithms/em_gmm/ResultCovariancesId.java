/* file: ResultCovariancesId.java */
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

package com.intel.daal.algorithms.em_gmm;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__RESULTCOVARIANCESID"></a>
 * @brief Available identifiers of results of the EM for GMM algorithm
 */
public final class ResultCovariancesId {
    private int _value;

    public ResultCovariancesId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int covariancesValue = 2;

    public static final ResultCovariancesId covariances = new ResultCovariancesId(
            covariancesValue); /*!< Covariances */
}
