/* file: ParameterMiniBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.optimization_solver.sgd.BaseParameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETERMINIBATCH"></a>
 * @brief ParameterMiniBatch of the SGD algorithm
 */
public class ParameterMiniBatch extends BaseParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context       Context to manage the parameter for SGD algorithm
     */
    public ParameterMiniBatch(DaalContext context) {
        super(context);
    }

    /**
    * Constructs the parameter for SGD algorithm
    * @param context                Context to manage the SGD algorithm
    * @param cParameterMiniBatch    Pointer to C++ implementation of the parameter
    */
    public ParameterMiniBatch(DaalContext context, long cParameterMiniBatch) {
        super(context, cParameterMiniBatch);
    }

    /**
    * Sets the numeric table of values of the conservative coefficient sequence
    * @param innerNIterations The numeric table of values of the conservative coefficient sequence
    */
    public void setInnerNIterations(long innerNIterations) {
        cSetInnerNIterations(this.cObject, innerNIterations);
    }

    /**
    * Returns the numeric table of values of the conservative coefficient sequence
    * @return The numeric table of values of the conservative coefficient sequence
    */
    public long getInnerNIterations() {
        return cGetInnerNIterations(this.cObject);
    }

    /**
    * Sets the numeric table of values of the conservative coefficient sequence
    * @param conservativeSequence The numeric table of values of the conservative coefficient sequence
    */
    public void setConservativeSequence(NumericTable conservativeSequence) {
        cSetConservativeSequence(this.cObject, conservativeSequence.getCObject());
    }

    /**
     * Gets the numeric table of values of the conservative coefficient sequence
     * @return The numeric table of values of the conservative coefficient sequence
     */
    public NumericTable getConservativeSequence() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetConservativeSequence(this.cObject));
    }

    private native void cSetInnerNIterations(long cObject, long innerNIterations);
    private native long cGetInnerNIterations(long cObject);
    private native void cSetConservativeSequence(long cObject, long conservativeSequence);
    private native long cGetConservativeSequence(long cObject);
}
/** @} */
