/* file: InternalOptionalDataId.java */
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
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;
import com.intel.daal.algorithms.optimization_solver.sgd.OptionalDataId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__INTERNALOPTIONALDATAID"></a>
 * @brief Available identifiers of InternalOptionalDataId objects for the algorithm
 */
public final class InternalOptionalDataId {
    private int _value;

    /**
     * Constructs the internal optional data object identifier using the provided value
     * @param value     Value corresponding to the internal optional data object identifier
     */
    public InternalOptionalDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the internal optional data object identifier
     * @return Value corresponding to the internal optional data object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int rngStateId = OptionalDataId.pastUpdateVector.getValue() + 1;

    public static final InternalOptionalDataId rngState = new InternalOptionalDataId(rngStateId); /*!< Memory block with random numbers generator state */
}
/** @} */
