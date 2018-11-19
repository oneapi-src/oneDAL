/* file: Prediction.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup base_algorithms
 * @{
 */
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
/** @} */
