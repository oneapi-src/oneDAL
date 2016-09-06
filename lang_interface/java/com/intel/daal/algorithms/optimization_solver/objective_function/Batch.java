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
 * @brief Contains classes for computing objective functions
 */
package com.intel.daal.algorithms.optimization_solver.objective_function;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__BATCH"></a>
 * @brief %Base interface for the Objective function algorithm in the batch processing mode
 * \n<a href="DAAL-REF-OBJECTIVE_FUNTION-ALGORITHM">Objective function algorithm description and usage models</a>
 */
public abstract class Batch extends AnalysisBatch {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the objective functions algorithm by copying input objects and parameters of
     * another objective functions algorithm
     * @param context  Context to manage the objective function algorithm
     */
    public Batch(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the objective functions algorithm by copying input objects and parameters of
     * another objective functions algorithm
     * @param context  Context to manage the objective function algorithm
     * @param other    An algorithm to be used as the source to initialize the input objects
     *                 and parameters of this algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
    }

    /**
     * Registers user-allocated memory to store the results of computing the Objective function
     * in the batch processing mode
     * @param result    Structure to store results of computing the Objective function
     */
    public void setResult(Result result) {
        cSetResult(cObject, result.getCObject());
    }

    /**
     * Returns the newly allocated Objective function algorithm
     * with a copy of input objects and parameters of this Objective function algorithm
     * @param context    Context to manage the Objective function algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Batch clone(DaalContext context);

    protected native void cSetResult(long cAlgorithm, long cResult);
    protected native long cGetResult(long cAlgorithm);
}
