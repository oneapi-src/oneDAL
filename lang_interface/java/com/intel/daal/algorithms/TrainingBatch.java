/* file: TrainingBatch.java */
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
 *  <a name="DAAL-CLASS-ALGORITHMS__TRAININGBATCH"></a>
 *  @brief Provides methods to train models that depend on the data provided in batch mode.
 *         For example, these methods enable training the linear regression model.
 *         Classes that implement specific algorithms of model training in batch mode are derived classes of the TrainingBatch class.
 *         The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class TrainingBatch extends Algorithm implements Disposable {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the training algorithm
     * @param context  Context to manage the training algorithm
     */
    public TrainingBatch(DaalContext context) {
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
     * Returns the newly allocated training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context  Context to manage the training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingBatch clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
/** @} */
