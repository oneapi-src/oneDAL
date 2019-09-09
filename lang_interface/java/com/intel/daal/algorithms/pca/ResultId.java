/* file: ResultId.java */
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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULTID"></a>
 * @brief Available types of results of the PCA algorithm
 */
public final class ResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int eigenValuesId  = 0;
    private static final int eigenVectorsId = 1;
    private static final int meansId = 2;
    private static final int variancesId = 3;

    public static final ResultId eigenValues  = new ResultId(eigenValuesId);  /*!< Eigenvalues */
    public static final ResultId eigenVectors = new ResultId(eigenVectorsId); /*!< Eigenvectors */
    public static final ResultId means = new ResultId(meansId); /*!< Means */
    public static final ResultId variances = new ResultId(variancesId); /*!< Variances */
}
/** @} */
