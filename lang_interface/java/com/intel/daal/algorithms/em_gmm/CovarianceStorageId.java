/* file: CovarianceStorageId.java */
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
 * @ingroup em_gmm
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__COVARIANCESTORAGEID"></a>
 * @brief Available identifiers of covariance types in the EM for GMM algorithm
 */
public final class CovarianceStorageId {
    private int _value;

    /**
     * Constructs the covariance type object identifier using the provided value
     * @param value     Value corresponding to the covariance type object identifier
     */
    public CovarianceStorageId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the covariance type object identifier
     * @return Value corresponding to the covariance type object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int fullValue      = 0;
    private static final int diagonalValue  = 1;

    public static final CovarianceStorageId full     = new CovarianceStorageId(fullValue);      /*!< Full */
    public static final CovarianceStorageId diagonal = new CovarianceStorageId(diagonalValue);  /*!< Diagonal */
}
/** @} */
