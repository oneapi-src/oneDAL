/* file: PredictionDistributed.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.services.Disposable;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PREDICTIONDISTRIBUTED"></a>
 *  \brief Provides prediction methods depending on the model such as linearregression.Model.
 *         The methods of the class support different computation modes: batch, distributed, and online(see @ref ComputeMode).
 *         Classes that implement specific algorithms of the model based data prediction are derived classes of the PredictionDistributed class.
 *         The class additionally provides virtual methods for validation of input and output parameters of the algorithms.
 */
public abstract class PredictionDistributed extends Algorithm implements Disposable {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the prediction algorithm in the distributed processing mode
     * @param context  Context to manage the prediction algorithm in the distributed processing mode
     */
    public PredictionDistributed(DaalContext context) {
        super(context);
    }

    /**
     * Computes prediction results based on the model
     * @return PredictionDistributed results
     */
    public PartialResult compute() {
        cCompute(this.cObject);
        return null;
    }

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
     * Returns the newly allocated prediction algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context  Context to manage the prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract PredictionDistributed clone(DaalContext context);

    private native void cCompute(long algAddr);
    private native void cFinalizeCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);
    private native void cCheckFinalizeComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
/** @} */
