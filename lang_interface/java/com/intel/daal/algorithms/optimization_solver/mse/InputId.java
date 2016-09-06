/* file: InputId.java */
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

package com.intel.daal.algorithms.optimization_solver.mse;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUTID"></a>
 * @brief Available identifiers of input objects for the MSE algorithm
 */
public final class InputId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /**
     * Constructs the input identifier for MSE algorithm
     * @param value Value of identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the identifier of input object
     * @return Value corresponding to the identifier
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
