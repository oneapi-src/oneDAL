/* file: Batch.java */
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
 * @brief Contains classes for the objective functions that could be represented as a sum of functions
 */
package com.intel.daal.algorithms.optimization_solver.sum_of_functions;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.optimization_solver.objective_function.Result;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Input;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Parameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__BATCH"></a>
 * @brief Computes the Sum of functions algorithm in the batch processing mode
 * \n<a href="DAAL-REF-SUM_OF_FUNCTIONS-ALGORITHM">Sum of functions algorithm description and usage models</a>
 *
 * @par References
 *      - Input class
 *      - Parameter class
 */
public abstract class Batch extends com.intel.daal.algorithms.optimization_solver.objective_function.Batch {
    public long cBatchIface;    /*!< Pointer to the inner implementation of the service callback functionality */
    public Input          input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the Sum of functions algorithm in the batch processing mode
     * @param context  Context to manage the Sum of functions algorithm
     * @param numberOfTerms Number of terms in the objectiove function that can be represent as sum
     */
    public Batch(DaalContext context, long numberOfTerms) {
        super(context);
        this.cBatchIface = cInitBatchIface(numberOfTerms);
    }

    /**
     * Constructs the Sum of functions algorithm by copying input objects and parameters of
     * another Sum of functions algorithm
     * @param context  Context to manage the Sum of functions algorithm
     * @param other    An algorithm to be used as the source to initialize the input objects
     *                 and parameters of this algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.cBatchIface = cInitBatchIface(other.parameter.getNumberOfTerms());
    }

    /**
     * Registers user-allocated memory to store the results of computing the Sum of functions
     * in the batch processing mode
     * @param result Structure to store results of computing the Sum of functions
     */
    public void setResult(Result result) {
        cSetResult(cBatchIface, result.getCObject());
    }

    /**
     * Return the result of the algorithm
     * @return Result of the algorithm
     */
    public Result getResult() {
        return new Result(getContext(), cGetResult(cBatchIface));
    }

    /**
     * Sets correspond pointers to the native side. Must be called in inherited class constructor after input and parameter initializations.
     */
    public void setPointersToIface() {
        cSetPointersToIface(cBatchIface, input.getCObject(), parameter.cObject);
    }

    /**
    * Releases the memory allocated for the native algorithm object
    */
    @Override
    public void dispose() {
        super.dispose();
        if (cBatchIface != 0) {
            cDispose(cBatchIface);
            cBatchIface = 0;
        }
    }

    /**
     * Returns the newly allocated Sum of functions algorithm
     * with a copy of input objects and parameters of this Sum of functions algorithm
     * @param context Context to manage the Sum of functions algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Batch clone(DaalContext context);

    private native void cDispose(long cBatchIface);
    private native long cInitBatchIface(long numberOfTerms);
    private native void cSetPointersToIface(long cBatchIface, long cInput, long cParameter);
}
