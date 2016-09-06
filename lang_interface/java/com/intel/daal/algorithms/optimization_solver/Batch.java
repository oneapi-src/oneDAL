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
 * @brief Contains classes for computing the optimization solvers
 */
package com.intel.daal.algorithms.optimization_solver;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__BATCH"></a>
 * @brief %Base interface for the Optimization solver algorithm in the batch processing mode
 * \n<a href="DAAL-REF-OPTIMIZATION_SOLVER-ALGORITHM">Optimization solver algorithm description and usage models</a>
 */
public abstract class Batch extends com.intel.daal.algorithms.AnalysisBatch {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the optimization solver algorithm in the batch processing mode
     * @param context  Context to manage the optimization solver algorithm
     */
    public Batch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated Optimization solver algorithm
     * with a copy of input objects and parameters of this Optimization solver algorithm
     * @param context    Context to manage the Optimization solver algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Batch clone(DaalContext context);
}
