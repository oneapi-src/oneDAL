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

package com.intel.daal.algorithms.optimization_solver.sum_of_functions;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__INPUTID"></a>
 * @brief Available identifiers of input objects for the Sum of functions algorithm
 */
public final class InputId {
    private int _value;

    /**
     * Constructs the input identifier for the sum functions algorithm
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

    public static final InputId argument = new InputId(argumentId); /*!< %Input argument table */
}
