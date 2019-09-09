/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup logistic_loss
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.logistic_loss;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__PARAMETER"></a>
 * @brief Parameters of the logistic loss objective function algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.sum_of_functions.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for the logistic loss objective function algorithm
     * @param context       Context to manage the logistic loss objective function algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the value of the interceptFlag flag
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

    private native boolean cGetInterceptFlag(long parAddr);
    private native void cSetInterceptFlag(long parAddr, boolean flag);
    private native float cGetPenaltyL1(long parAddr);
    private native void cSetPenaltyL1(long parAddr, float value);
    private native float cGetPenaltyL2(long parAddr);
    private native void cSetPenaltyL2(long parAddr, float value);
}
/** @} */
