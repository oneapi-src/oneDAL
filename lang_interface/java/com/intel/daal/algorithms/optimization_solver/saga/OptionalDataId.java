/* file: OptionalDataId.java */
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
 * @ingroup saga
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.saga;

/**
 * <a name="DAAL-CLASS-ALGORITHM__OPTIMIZATION_SOLVER__SAGA__OPTIONALDATAID"></a>
 * @brief Available identifiers of optional data objects for the iterative algorithm
 */
public final class OptionalDataId {
    private int _value;

    /**
     * Constructs the optional data object identifier using the provided value
     * @param value     Value corresponding to the optional data object identifier
     */
    public OptionalDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional data object identifier
     * @return Value corresponding to the optional data object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int gradientsTableId = 0;

    public static final OptionalDataId gradientsTable = new OptionalDataId(gradientsTableId); /*!< %Accumulated sum of squares of corresponding gradient's coordinate values */
}
/** @} */
