/* file: Prediction.java */
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
 *  <a name="DAAL-CLASS-ALGORITHMS__PREDICTION"></a>
 *  \brief Provides prediction methods depending on the model such as linearregression.Model.
 *         The methods of the class support different computation modes: batch, distributed, and online(see @ref ComputeMode).
 *         Classes that implement specific algorithms of the model based data prediction are derived classes of the Prediction class.
 *         The class additionally provides virtual methods for validation of input and output parameters of the algorithms.
 */
public abstract class Prediction extends Algorithm implements Disposable {

    /**
     * Constructs the prediction algorithm
     * @param context  Context to manage the prediction algorithm
     */
    public Prediction(DaalContext context) {
        super(context);
    }

    /**
     * Computes prediction results based on the model
     * @return Prediction results
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
     * Returns the newly allocated prediction algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context  Context to manage the prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Prediction clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
