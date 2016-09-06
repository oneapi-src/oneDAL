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
 * @brief Contains classes for computing iterative solver algorithm
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__PARAMETER"></a>
 * @brief Parameter of the iterative solver algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameter for iterative solver algorithm
     * @param context       Context to manage the iterative solver algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Sets objective function represented as sum of functions
    * @param function Objective function represented as sum of functions
    */
    public void setFunction(Batch function) {
        _function = function;
        cSetFunction(this.cObject, function.cBatchIface);
    }

    /**
     * Gets objective function represented as sum of functions
     * @return Objective function represented as sum of functions
     */
    public Batch getFunction() {
        return _function;
    }

    /**
    * Sets the maximal number of iterations of the algorithm
    * @param nIterations The maximal number of iterations of the algorithm
    */
    public void setNIterations(long nIterations) {
        cSetNIterations(this.cObject, nIterations);
    }

    /**
     * Gets the maximal number of iterations of the algorithm
     * @return The maximal number of iterations of the algorithm
     */
    public long getNIterations() {
        return cGetNIterations(this.cObject);
    }

    /**
    * Sets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    * @param accuracyThreshold The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Gets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * @return The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the optionalResultRequired flag
     * @param flag    The flag. If true, optional result is calculated
     */
    public void setOptionalResultRequired(boolean flag) {
        cSetOptionalResultRequired(this.cObject, flag);
    }

    /**
     * Gets the optionalResultRequired flag
     * @return The flag
     */
    public boolean getOptionalResultRequired() {
        return cGetOptionalResultRequired(this.cObject);
    }

    private Batch _function;

    private native void cSetFunction(long parAddr, long function);

    private native void cSetNIterations(long parAddr, long nIterations);
    private native long cGetNIterations(long parAddr);

    private native void cSetAccuracyThreshold(long parAddr, double accuracyThreshold);
    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetOptionalResultRequired(long parAddr, boolean flag);
    private native boolean cGetOptionalResultRequired(long parAddr);

}
