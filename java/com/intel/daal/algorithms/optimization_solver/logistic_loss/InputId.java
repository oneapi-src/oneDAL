/* file: InputId.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup logistic_loss
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.logistic_loss;

import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__INPUTID"></a>
 * @brief Available identifiers of input objects for the logistic loss objective function algorithm
 */
public final class InputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int argumentId = 0;
    private static final int dataId = 1;
    private static final int dependentVariablesId = 2;

    public static final InputId argument           = new InputId(argumentId);           /*!< %Input argument table */
    public static final InputId data               = new InputId(dataId);               /*!< %Input data table */
    public static final InputId dependentVariables = new InputId(dependentVariablesId); /*!< %Input dependent variables table */
}
/** @} */
