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

/**
 * @brief Contains classes for computing Adagrad algorithm
 */
package com.intel.daal.algorithms.optimization_solver.adagrad;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__PARAMETER"></a>
 * @brief Parameter of the Adagrad algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameter for Adagrad algorithm
     * @param context       Context to manage the Adagrad algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for Adagrad algorithm
     * @param context    Context to manage the Adagrad algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     * @param batchIndices The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     */
    public void setBatchIndices(NumericTable batchIndices) {
        cSetBatchIndices(this.cObject, batchIndices.getCObject());
    }

    /**
     * Gets the numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     * @return The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     */
    public NumericTable getBatchIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBatchIndices(this.cObject));
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
     * Sets the numeric table that contains value of the learning rate
     * @param learningRate The numeric table that contains value of the learning rate
     */
    public void setLearningRate(NumericTable learningRate) {
        cSetLearningRate(this.cObject, learningRate.getCObject());
    }

    /**
     * Gets the numeric table that contains value of the learning rate
     * @return The numeric table that contains value of the learning rate
     */
    public NumericTable getLearningRate() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetLearningRate(this.cObject));
    }

    /**
     * Sets the value needed to avoid degenerate cases in square root computing
     * @param degenerateCasesThreshold The value needed to avoid degenerate cases in square root computing
     */
    public void setDegenerateCasesThreshold(double degenerateCasesThreshold) {
        cSetDegenerateCasesThreshold(this.cObject, degenerateCasesThreshold);
    }

    /**
     * Retrieves the value needed to avoid degenerate cases in square root computing
     * @return The value needed to avoid degenerate cases in square root computing
     */
    public double getDegenerateCasesThreshold() {
        return cGetDegenerateCasesThreshold(this.cObject);
    }

    /**
    * Sets the seed for random generation of 32 bit integer indices of terms in the objective function.
    * @param seed The seed for random generation of 32 bit integer indices of terms in the objective function.
    */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * Gets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @return The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    private native void cSetBatchIndices(long parAddr, long batchIndicesAddr);
    private native long cGetBatchIndices(long parAddr);

    private native void cSetBatchSize(long parAddr, long batchSize);
    private native long cGetBatchSize(long parAddr);

    private native void cSetLearningRate(long parAddr, long learningRateAddr);
    private native long cGetLearningRate(long parAddr);

    private native void cSetDegenerateCasesThreshold(long parAddr, double degenerateCasesThreshold);
    private native double cGetDegenerateCasesThreshold(long parAddr);

    private native void cSetSeed(long parAddr, int seed);
    private native int cGetSeed(long parAddr);
}
