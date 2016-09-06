/* file: AnalysisBatch.java */
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

package com.intel.daal.algorithms;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISBATCH"></a>
 * @brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates in batch mode.
 *        Classes that implement specific algorithms of the data analysis in batch mode are derived classes of the AnalysisBatch class.
 *        The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class AnalysisBatch extends Algorithm {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the algorithm
     * @param context  Context to manage the algorithm
     */
    public AnalysisBatch(DaalContext context) {
        super(context);
    }

    /**
     * Computes final results of the algorithm in batch mode.
     * @return Final results of the algorithm
     */
    public Result compute() {
        cCompute(this.cObject);
        return null;
    }

    /**
     * Validates parameters of the compute method
     */
    @Override
    public void checkComputeParams() {
        cCheckComputeParams(this.cObject);
    }

    /**
     * Releases memory allocated for the native algorithm object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    /**
     * Returns the newly allocated algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context  Context to manage the algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract AnalysisBatch clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
