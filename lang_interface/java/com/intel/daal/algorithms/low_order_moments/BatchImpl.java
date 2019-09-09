/* file: BatchImpl.java */
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
 * @ingroup low_order_moments_batch
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCHIFACE"></a>
 * @brief %Base interface for the low order moments algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * @par Enumerations
 *      - @ref Method   Computation methods of the low order moments algorithm
 *      - @ref InputId  Identifiers of input objects
 *      - @ref ResultId Identifiers of the results
 *
 * @par References
 *      - Input class
 *      - Parameter class
 *      - Result class
 */
public abstract class BatchImpl extends AnalysisBatch {
    public long cBatchImpl; /*!< Pointer to the inner implementation of the service callback functionality */
    public Input      input; /*!< %Input objects for the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    protected Precision prec; /*!< Precision of computations */


    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__LOW_ORDER_MOMENTS__BATCHIFACE__BATCHIFACE"></a>
     * @param context  Context to manage the low order moments algorithm
     */
    public BatchImpl(DaalContext context) {
        super(context);
        this.cBatchImpl = cInitBatchImpl();
    }

    /**
     * Computes the low order moments algorithm in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of computing the low order moments algorithm
     * in the batch processing mode
     * @param result    Structure to store results of computing the low order moments algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Releases the memory allocated for the native algorithm object
     */
    @Override
    public void dispose() {
        if (this.cBatchImpl != 0) {
            cDispose(this.cBatchImpl);
            this.cBatchImpl = 0;
        }
        super.dispose();
    }

    /**
     * Returns the newly allocated low order moments algorithm
     * with a copy of input objects and parameters of this algorithm for low order moments computation
     * @param context    Context to manage the low order moments algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract BatchImpl clone(DaalContext context);

    private native long cInitBatchImpl();
    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cDispose(long cBatchImpl);
}
/** @} */
