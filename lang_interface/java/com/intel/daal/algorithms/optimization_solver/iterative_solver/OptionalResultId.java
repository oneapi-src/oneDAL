/* file: OptionalResultId.java */
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
 * @ingroup iterative_solver
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALRESULTID"></a>
 * @brief Available result identifiers for the iterative solver algorithm
 */
public final class OptionalResultId {
    private int _value;

    /**
     * Constructs the optional result object identifier using the provided value
     * @param value     Value corresponding to the optional result object identifier
     */
    public OptionalResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional result object identifier
     * @return Value corresponding to the optional result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int optionalResultId = 2;

    public static final OptionalResultId optionalResult  = new OptionalResultId(optionalResultId); /*!< Algorithm-specific result data */
}
/** @} */
