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
 * @ingroup adagrad
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.adagrad;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.OptionalArgument;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__INPUT"></a>
 * @brief %Input objects for the Adagrad algorithm
 */
public class Input extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input for the Adagrad algorithm
     * @param context       Context to manage the input for the Adagrad algorithm
     */
    public Input(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the input for the Adagrad algorithm
     * @param context       Context to manage the Adagrad algorithm
     * @param cInput        Pointer to C++ implementation of the input
     */
    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Sets an optional input data for the Adagrad algorithm
     * @param id    Identifier of the optional data object
     * @param val   The optional data object
     */
    public void set(OptionalDataId id, NumericTable val) {
        if (id != OptionalDataId.gradientSquareSum) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetOptionalData(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an optional input data for the Adagrad algorithm
     * @param id Identifier of the optional data object
     * @return   %Optional data object that corresponds to the given identifier
     */
    public NumericTable get(OptionalDataId id) {
        if (id != OptionalDataId.gradientSquareSum) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetOptionalData(cObject, id.getValue()));
    }

    protected native void cSetOptionalData(long cInput, int id, long ntAddr);
    protected native long cGetOptionalData(long cInput, int id);
}
/** @} */
