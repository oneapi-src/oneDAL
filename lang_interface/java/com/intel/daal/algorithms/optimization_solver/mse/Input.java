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
 * @ingroup mse
 * @{
 */
/**
 * @brief Contains classes for computing the mse algorithm
 */
package com.intel.daal.algorithms.optimization_solver.mse;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUT"></a>
 * @brief %Input objects for the MSE algorithm
 */
public class Input extends com.intel.daal.algorithms.optimization_solver.sum_of_functions.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input for MSE algorithm
     * @param context       Context to manage the MSE algorithm
     * @param cInput        Pointer to C++ implementation of the input
     */
    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Sets an input object for the MSE algorithm
     * @param id    Identifier of the input object
     * @param val   The input object
     */
    public void set(InputId id, NumericTable val) {
        if (id != InputId.argument && id != InputId.data && id != InputId.dependentVariables) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for the MSE algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id != InputId.argument && id != InputId.data && id != InputId.dependentVariables) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
    }

    // private native long cInit(long algAddr, int prec, int method);
    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInputTable(long cObject, int id);
}
/** @} */
