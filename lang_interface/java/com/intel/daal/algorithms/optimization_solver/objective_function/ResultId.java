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

package com.intel.daal.algorithms.optimization_solver.objective_function;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTID"></a>
 * @brief Available result identifiers for the objective funtion algorithm
 */
public final class ResultId {
    private int _value;

    /**
     * Constructs the result identifier for objective function algorithm
     * @param value Value of identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
    * Returns the value corresponding to the identifier of result object
    * @return Value corresponding to the identifier
    */
    public int getValue() {
        return _value;
    }

    private static final int resultCollectionId = 0;

    public static final ResultId resultCollection  = new ResultId(resultCollectionId);  /*!< Collection of the objective function results. */
}
