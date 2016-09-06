/* file: AnalysisDistributed.java */
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
import com.intel.daal.services.Disposable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISDISTRIBUTED"></a>
 * @brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates in distributed mode.
 *        Classes that implement specific algorithms of the data analysis in distributed mode are derived classes of the AnalysisDistributed class.
 *        The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class AnalysisDistributed extends Algorithm implements Disposable {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the algorithm
     * @param context  Context to manage the algorithm
     */
    public AnalysisDistributed(DaalContext context) {
        super(context);
    }

    /**
     * Computes partial results of the algorithm in distributed mode
     * @return Partial results of the algorithm
     */
    public PartialResult compute() {
        cCompute(this.cObject);
        return null;
    }

    /**
     * Computes final results of the algorithm using partial results in distributed mode.
     * @return Final results of the algorithm
     */
    public Result finalizeCompute() {
        cFinalizeCompute(this.cObject);
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
     * Validates parameters of the finalizeCompute method
     */
    public void checkFinalizeComputeParams() {
        cCheckFinalizeComputeParams(this.cObject);
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
    public abstract AnalysisDistributed clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cFinalizeCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cCheckFinalizeComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
