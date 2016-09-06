/* file: Parameter.java */
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

package com.intel.daal.algorithms.optimization_solver.lbfgs;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__PARAMETER"></a>
 * @brief Parameters of the LBFGS algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameters of LBFGS algorithm
     * @param context Context to manage the parameters of LBFGS algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
    * Constructs the parameter for LBFGS algorithm
    * @param context    Context to manage the LBFGS algorithm
    * @param cObject    Pointer to C++ implementation of the parameter
    */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the memory parameter of LBFGS algorithm. Which is the maximum number of correction pairs
     * that define the approximation of inverse Hessian matrix
     * @param m The memory parameter of LBFGS algorithm
     */
    public void setM(long m) {
        cSetM(this.cObject, m);
    }

    /**
     * Gets the memory parameter of LBFGS algorithm. Which is the maximum number of correction pairs
     * that define the approximation of inverse Hessian matrix
     * @return The memory parameter of LBFGS algorithm
     */
    public long getM() {
        return cGetM(this.cObject);
    }

    /**
     * Sets the number of iterations between the curvature estimates calculations
     * @param L The number of iterations between the curvature estimates calculations
     */
    public void setL(long L) {
        cSetL(this.cObject, L);
    }

    /**
     * Gets the number of iterations between the curvature estimates calculations
     * @return The number of iterations between the curvature estimates calculations
     */
    public long getL() {
        return cGetL(this.cObject);
    }

    /**
     * Sets the number of terms of objective function that used to compute the stochastic gradient
     * @param batchSize The number of terms of objective function
     */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(this.cObject, batchSize);
    }

    /**
     * Gets the number of terms of objective function that used to compute the stochastic gradient
     * @return The number of terms of objective function
     */
    public long getBatchSize() {
        return cGetBatchSize(this.cObject);
    }

    /**
     * Sets the numeric table of size nIterations x batchSize that represent indices
     * that will be used instead of random values for the stochastic gradient computations.
     * If no indices are provided, the implementation will generate random indices.
     * @param batchIndices The numeric table that represents 32 bit integer indices of terms of the objective function
     */
    public void setBatchIndices(NumericTable batchIndices) {
        cSetBatchIndices(this.cObject, batchIndices.getCObject());
    }

    /**
     * Gets the numeric table that represents 32 bit integer indices of terms of the objective function.
     * @return The numeric table that represents 32 bit integer indices of terms of the objective function
     */
    public NumericTable getBatchIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBatchIndices(this.cObject));
    }

    /**
     * Sets the number of observations to compute the sub-sampled Hessian for correction pairs computation
     * @param batchSize The number of observations to compute the sub-sampled Hessian for correction pairs computation
     */
    public void setCorrectionPairBatchSize(long batchSize) {
        cSetCorrectionPairBatchSize(this.cObject, batchSize);
    }

    /**
     * Gets the number of observations to compute the sub-sampled Hessian for correction pairs computation
     * @return The number of observations to compute the sub-sampled Hessian for correction pairs computation
     */
    public long getCorrectionPairBatchSize() {
        return cGetCorrectionPairBatchSize(this.cObject);
    }

    /**
     * Sets the numeric table of size (nIterations / L) x correctionPairBatchSize that represent indices
     * that will be used instead of random values for the sub-sampled Hessian matrix computations.
     * If no indices are provided, the implementation will generate random indices.
     * @param batchIndices The numeric table that represents 32 bit integer indices of terms of the objective function
     */
    public void setCorrectionPairBatchIndices(NumericTable batchIndices) {
        cSetCorrectionPairBatchIndices(this.cObject, batchIndices.getCObject());
    }

    /**
     * Gets the numeric table that represents 32 bit integer indices of terms of the objective function.
     * @return The numeric table that represents 32 bit integer indices of terms of the objective function
     */
    public NumericTable getCorrectionPairBatchIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetCorrectionPairBatchIndices(this.cObject));
    }

    /**
     * Sets the numeric table that contains values of the step-length sequence
     * @param stepLengthSequence The numeric table that contains values of the step-length sequence
     */
    public void setStepLengthSequence(NumericTable stepLengthSequence) {
        cSetStepLengthSequence(this.cObject, stepLengthSequence.getCObject());
    }

    /**
     * Gets the numeric table that contains values of the step-length sequence
     * @return The numeric table that contains values of the step-length sequence
     */
    public NumericTable getStepLengthSequence() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetStepLengthSequence(this.cObject));
    }

    /**
     * Sets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @param seed The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public void setSeed(long seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * Gets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @return The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public long getSeed() {
        return cGetSeed(this.cObject);
    }

    private native void cSetM(long parAddr, long m);
    private native long cGetM(long parAddr);

    private native void cSetL(long parAddr, long L);
    private native long cGetL(long parAddr);

    private native void cSetBatchSize(long parAddr, long batchSize);
    private native long cGetBatchSize(long parAddr);

    private native void cSetBatchIndices(long parAddr, long batchIndices);
    private native long cGetBatchIndices(long parAddr);

    private native void cSetCorrectionPairBatchSize(long parAddr, long batchSize);
    private native long cGetCorrectionPairBatchSize(long parAddr);

    private native void cSetCorrectionPairBatchIndices(long parAddr, long batchIndices);
    private native long cGetCorrectionPairBatchIndices(long parAddr);

    private native void cSetStepLengthSequence(long parAddr, long stepLengthSequence);
    private native long cGetStepLengthSequence(long parAddr);

    private native void cSetSeed(long parAddr, long seed);
    private native long cGetSeed(long parAddr);

}
