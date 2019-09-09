/* file: OptionalInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALINPUTID"></a>
 * @brief Available identifiers of optional input objects for the iterative algorithm
 */
public final class OptionalInputId {
    private int _value;

    /**
     * Constructs the optional input object identifier using the provided value
     * @param value     Value corresponding to the optional input object identifier
     */
    public OptionalInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional input object identifier
     * @return Value corresponding to the optional input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int optionalArgumentId = 1;

    public static final OptionalInputId optionalArgument = new OptionalInputId(optionalArgumentId); /*!< %Algorithm-specific input data */
}
/** @} */
