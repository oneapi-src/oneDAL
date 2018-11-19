/* file: AnalysisOnline.java */
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

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.services.Disposable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISONLINE"></a>
 * @brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates in the online processing mode.
 *        Classes that implement specific algorithms of the data analysis in the online processing mode are derived classes of the AnalysisOnline class.
 *        The class additionally provides methods for validation of input and output parameters of the algorithms.
 */
public abstract class AnalysisOnline extends Algorithm implements Disposable {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the algorithm in the online processing mode
     * @param context  Context to manage the algorithm in the online processing mode
     */
    public AnalysisOnline(DaalContext context) {
        super(context);
    }

    /**
     * Computes partial results of the algorithm in the online processing mode
     * @return Partial results of the algorithm
     */
    public PartialResult compute() {
        cCompute(this.cObject);
        return null;
    }

    /**
     * Computes final results of the algorithm using partial results in the online processing mode.
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
    public abstract AnalysisOnline clone(DaalContext context);

    private native void cCompute(long algAddr);

    private native void cFinalizeCompute(long algAddr);

    private native void cCheckComputeParams(long algAddr);

    private native void cCheckFinalizeComputeParams(long algAddr);

    private native void cDispose(long algAddr);
}
/** @} */
