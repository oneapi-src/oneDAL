/* file: TrainingParameter.java */
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
 * @ingroup logistic_regression_training
 * @{
 */
/**
 * \brief Logistic regression algorithm parameters
 */
package com.intel.daal.algorithms.logistic_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAININGPARAMETER"></a>
 * @brief Logistic regression training algorithm parameters
 */
public class TrainingParameter extends com.intel.daal.algorithms.classifier.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingParameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the interceptFlag flag that enables or disables the computation
     * of the beta0 coefficient in the logistic regression equation
     * @param flag
     */
    public void setInterceptFlag(boolean flag) {
        cSetInterceptFlag(this.cObject, flag);
    }

    /**
     * Returns the value of the interceptFlag flag
     * @return Flag
     */
    public boolean getInterceptFlag() {
        return cGetInterceptFlag(this.cObject);
    }

    /**
     * Returns L1 regularization coefficient
     * @return PenaltyL1
     */
    public float getPenaltyL1() {
        return cGetPenaltyL1(this.cObject);
    }

    /**
     * Sets L1 regularization coefficient
     * @param value
     */
    public void setPenaltyL1(float value) {
        cSetPenaltyL1(this.cObject, value);
    }

    /**
     * Returns L2 regularization coefficient
     * @return PenaltyL2
     */
    public float getPenaltyL2() {
        return cGetPenaltyL2(this.cObject);
    }

    /**
     * Sets L2 regularization coefficient
     * @param value
     */
    public void setPenaltyL2(float value) {
        cSetPenaltyL2(this.cObject, value);
    }

    /**
     * Sets the optimization solver to be used by the algorithm
     * @param optimizationSolver optimization solver to be used by the algorithm
     */
    public void setOptimizationSolver(com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch optimizationSolver) {
        cSetOptimizationSolver(cObject, optimizationSolver.cObject);
    }

    /**
     *  Gets the optimization solver used by the algorithm
     */
    public com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch getOptimizationSolver() {
        return new com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch(getContext(), cGetOptimizationSolver(cObject));
    }

    private native boolean cGetInterceptFlag(long parAddr);
    private native void cSetInterceptFlag(long parAddr, boolean flag);
    private native float cGetPenaltyL1(long parAddr);
    private native void cSetPenaltyL1(long parAddr, float value);
    private native float cGetPenaltyL2(long parAddr);
    private native void cSetPenaltyL2(long parAddr, float value);
    private native void cSetOptimizationSolver(long cObject, long cOptimizationSolverObject);
    private native long cGetOptimizationSolver(long cObject);
}
/** @} */
