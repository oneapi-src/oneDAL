/* file: TrainParameter.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
 * @ingroup elastic_net
 * @{
 */
/**
 * \brief Contains classes for computing the result of the elastic net algorithm
 */
package com.intel.daal.algorithms.elastic_net;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__TRAINPARAMETER"></a>
 * @brief Elastic net algorithm parameters
 */
public class TrainParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainParameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the numeric table that represents L1 elastic net parameters. If no L1 elastic net parameters are provided,
     * the implementation will generate L1 elastic net parameters equal to 1.
     * @param penaltyL1 The numeric table that represents L1 elastic net parameters
     */
    public void setPenaltyL1(NumericTable penaltyL1) {
        cSetPenaltyL1(this.cObject, penaltyL1.getCObject());
    }

    /**
     * Sets the numeric table that represents L2 elastic net parameters. If no L2 elastic net parameters are provided,
     * the implementation will generate L2 elastic net parameters equal to 1.
     * @param penaltyL2 The numeric table that represents L2 elastic net parameters
     */
    public void setPenaltyL2(NumericTable penaltyL2) {
        cSetPenaltyL2(this.cObject, penaltyL2.getCObject());
    }

    /**
     * Retrieves the numeric table that represents L1 elastic net parameters. If no L1 elastic net parameters are provided,
     * the implementation will generate L1 elastic net parameters equal to 1.
     * @return The numeric table that represents L1 elastic net parameters.
     */
    public NumericTable getPenaltyL1() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPenaltyL1(this.cObject));
    }

    /**
     * Retrieves the numeric table that represents L2 elastic net parameters. If no L2 elastic net parameters are provided,
     * the implementation will generate L2 elastic net parameters equal to 1.
     * @return The numeric table that represents L2 elastic net parameters.
     */
    public NumericTable getPenaltyL2() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPenaltyL2(this.cObject));
    }

    /**
     * Sets the interceptFlag flag that enables or disables the computation
     * of the beta0 coefficient in the elastic net equation
     * @param flag
     */
    public void setInterceptFlag(boolean flag) {
        cSetInterceptFlag(this.cObject, flag);
    }

    /**
     * Returns the value of the interceptFlag flag
     */
    public boolean getInterceptFlag() {
        return cGetInterceptFlag(this.cObject);
    }

    /**
     * Sets the Data Use in Computation flag that allows modification of input data
     * @param flag
     */
    public void setDataUseInComputation(int flag) {
        cSetDataUseInComputation(this.cObject, flag);
    }

    /**
     * Returns the value of the Data Use in Computation flag
     */
    public int getDataUseInComputation() {
        return cGetDataUseInComputation(this.cObject);
    }

    /**
     * Sets the optional result to compute flag
     * @param optResult
     */
    public void setOptResultToCompute(int optResult) {
        cSetOptResultToCompute(this.cObject, optResult);
    }

    /**
     * Returns the value of the Data Use in Computation flag
     */
    public int getOptResultToCompute() {
        return cGetOptResultToCompute(this.cObject);
    }

    /**
     * Sets the optimization solver to be used by the algorithm
     * @param optimizationSolver optimization solver to be used by the algorithm
     */
    public void setOptimizationSolver(com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch optimizationSolver) {
        cSetOptimizationSolver(this.cObject, optimizationSolver.cObject);
    }

    /**
     *  Gets the optimization solver used in the algorithm
     */
    public com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch getOptimizationSolver() {
        return new com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch(getContext(), cGetOptimizationSolver(this.cObject));
    }

    private native void cSetPenaltyL1(long parAddr, long penaltyL1);
    private native void cSetPenaltyL2(long parAddr, long penaltyL2);
    private native long cGetPenaltyL1(long parAddr);
    private native long cGetPenaltyL2(long parAddr);
    private native void cSetInterceptFlag(long parAddr, boolean interceptFlag);
    private native boolean cGetInterceptFlag(long parAddr);
    private native void cSetDataUseInComputation(long parAddr, int flag);
    private native int cGetDataUseInComputation(long parAddr);
    private native void cSetOptResultToCompute(long parAddr, int optResult);
    private native int  cGetOptResultToCompute(long parAddr);
    private native void cSetOptimizationSolver(long cObject, long cOptimizationSolverObject);
    private native long cGetOptimizationSolver(long cObject);
}
/** @} */
