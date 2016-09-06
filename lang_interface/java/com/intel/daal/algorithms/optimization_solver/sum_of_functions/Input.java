/* file: Input.java */
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__INPUT"></a>
 * @brief %Input objects for the Sum of functions algorithm
 */
public class Input extends com.intel.daal.algorithms.optimization_solver.objective_function.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private long cCreatedInput; /*!< Pointer to C++ interface implementation of the input */

    /**
     * Constructs the input for the sum of functions algorithm
     * @param context       Context to manage the sum of functions algorithm
     * @param cInput        Pointer to C++ implementation of the input
     */
    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Constructs the input for the sum of functions algorithm
     * @param context       Context to manage the sum of functions algorithm
     */
    public Input(DaalContext context) {
        super(context);
        this.cCreatedInput = cCreateInput();
        this.cObject = this.cCreatedInput;
    }

    /**
     * Sets an input object for the Sum of functions algorithm
     * @param id    Identifier of the input object
     * @param val   The input object
     */
    public void set(InputId id, NumericTable val) {
        if (id != InputId.argument) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for the Sum of functions algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id != InputId.argument) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
    }

    /**
     * Sets input pointer for algorithm in native side
     * @param cInput     The address of the native input object
     * @param cAlgorithm The address of the native algorithm object
     */
    public void setCInput(long cInput, long cAlgorithm) {
        this.cObject = cInput;
        cSetCInput(this.cObject, cAlgorithm);
    }

    /**
    * Releases memory allocated for the native parameter object
    */
    @Override
    public void dispose() {
        if(this.cCreatedInput != 0) {
            cInputDispose(this.cCreatedInput);
            this.cCreatedInput = 0;
        }
    }

    private native void cSetInput(long cInput, int id, long ntAddr);
    private native long cGetInput(long cInput, int id);
    private native void cSetCInput(long cObject, long cAlgorithm);
    private native long cCreateInput();
    private native void cInputDispose(long cCreatedInput);
}
