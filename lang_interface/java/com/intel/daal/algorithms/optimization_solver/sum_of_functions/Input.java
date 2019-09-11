/* file: Input.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup sum_of_functions
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sum_of_functions;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
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
        LibUtils.loadLibrary();
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
     * @param context       Context to manage the input for the sum of functions algorithm
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
/** @} */
