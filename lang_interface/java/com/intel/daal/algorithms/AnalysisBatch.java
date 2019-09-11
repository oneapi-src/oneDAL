/* file: AnalysisBatch.java */
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISBATCH"></a>
 * @brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates in batch processing mode.
 *        Classes that implement specific algorithms of the data analysis in batch processing mode are derived classes of the AnalysisBatch class.
 *        The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class AnalysisBatch extends Algorithm {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the algorithm in the batch processing mode
     * @param context  Context to manage the algorithm in the batch processing mode
     */
    public AnalysisBatch(DaalContext context) {
        super(context);
    }

    /**
     * Computes final results of the algorithm in the batch processing mode.
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
/** @} */
