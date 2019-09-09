/* file: TrainingDistributed.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.services.DaalContext;
import com.intel.daal.services.Disposable;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__TRAININGDISTRIBUTED"></a>
 *  @brief Provides methods to train models that depend on the data provided in the distributed processing mode.
 *         For example, these methods enable training the linear regression model.
 *         Classes that implement specific algorithms of model training in the distributed processing mode are derived classes of the TrainingDistributed class.
 *         The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class TrainingDistributed extends Algorithm implements Disposable {

    /**
     * Constructs the training algorithm in the distributed processing mode
     * @param context  Context to manage the training algorithm in the distributed processing mode
     */
    public TrainingDistributed(DaalContext context) {
        super(context);
    }

    /**
     * Computes partial results of the algorithm in the distributed processing mode
     * @return Partial results of the algorithm
     */
    public PartialResult compute() {
        cCompute(this.cObject);
        return null;
    }

    /**
     * Computes final results of the algorithm using partial results in the distributed processing mode.
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
     * Returns the newly allocated training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context  Context to manage the training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingDistributed clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cFinalizeCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cCheckFinalizeComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
/** @} */
