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
 * @ingroup em_gmm_compute
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__RESULTID"></a>
 * @brief Available identifiers of results of the EM for GMM algorithm
 */
public final class ResultId {
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

    @Native private static final int weightsValue      = 0;
    @Native private static final int meansValue        = 1;
    @Native private static final int goalFunctionValue = 3;
    @Native private static final int nIterationsValue  = 4;

    public static final ResultId weights      = new ResultId(weightsValue);      /*!< Weights */
    public static final ResultId means        = new ResultId(meansValue);        /*!< Means */
    public static final ResultId goalFunction = new ResultId(goalFunctionValue); /*!< Goal function value */
    public static final ResultId nIterations  = new ResultId(nIterationsValue);  /*!< Number of iterations */
}
/** @} */
