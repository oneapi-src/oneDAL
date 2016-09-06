/* file: ParameterMiniBatch.java */
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

package com.intel.daal.algorithms.optimization_solver.sgd;

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
        System.loadLibrary("JavaAPI");
    }

    /**
    * Constructs the parameter for SGD algorithm
    * @param context       Context to manage the SGD algorithm
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
    * Sets the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @param batchSize The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(this.cObject, batchSize);
    }

    /**
    * Returns the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @return The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public long getBatchSize() {
        return cGetBatchSize(this.cObject);
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

    private native void cSetBatchSize(long cObject, long batchSize);
    private native long cGetBatchSize(long cObject);
    private native void cSetInnerNIterations(long cObject, long innerNIterations);
    private native long cGetInnerNIterations(long cObject);
    private native void cSetConservativeSequence(long cObject, long conservativeSequence);
    private native long cGetConservativeSequence(long cObject);
}
