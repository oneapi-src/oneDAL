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
 * @ingroup covariance_batch
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHIFACE"></a>
 * @brief %Base interface for the correlation or variance-covariance matrix algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation or variance-covariance matrix algorithm description and usage models</a> -->
 *
 * @tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * @tparam method           Computation method, @ref daal::algorithms::covariance::Method
 *
 * @par Enumerations
 *      - @ref Method   Computation methods of the correlation or variance-covariance matrix algorithm
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
    public Parameter  parameter; /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    protected Precision prec; /*!< Precision of computations */


    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the correlation or variance-covariance matrix algorithm in the batch processing mode
     * @param context  Context to manage the correlation or variance-covariance matrix algorithm in the batch processing mode
     */
    public BatchImpl(DaalContext context) {
        super(context);
        this.cBatchImpl = cInitBatchImpl();
    }

    /**
     * Computes the correlation or variance-covariance matrix in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of computing the correlation or variance-covariance matrix
     * in the batch processing mode
     * @param result    Structure to store results of computing the correlation or variance-covariance matrix
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
     * Returns the newly allocated correlation or variance-covariance matrix algorithm
     * with a copy of input objects and parameters of this algorithm for correlation or variance-covariance
     * matrix computation
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract BatchImpl clone(DaalContext context);

    private native long cInitBatchImpl();
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);
    private native void cDispose(long cBatchImpl);
}
/** @} */
