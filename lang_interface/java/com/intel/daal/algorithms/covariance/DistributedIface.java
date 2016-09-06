/* file: DistributedIface.java */
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

/**
 * @brief Contains classes for computing the correlation or variance-covariance matrix
 * in the distributed processing mode
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDIFACE"></a>
 * @brief %Base interface for the correlation or variance-covariance matrix algorithm in the distributed processing mode
 * \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation or variance-covariance matrix algorithm description and usage models</a>
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
public abstract class DistributedIface extends AnalysisDistributed {
    public long                        cDistributedIface; /*!< Pointer to the inner implementation of the service callback functionality */
    public DistributedStep2MasterInput input; /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    protected PartialResult            partialResult;     /*!< Partial result of the algorithm */
    public Precision                   prec; /*!< Precision of computations */
    public Parameter parameter; /*!< Algorithm parameter */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__COVARIANCE__DISTRIBUTEDIFACE__DISTRIBUTEDIFACE"></a>
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     */
    public DistributedIface(DaalContext context) {
        super(context);
        this.cDistributedIface = cInitDistributedIface();
    }


    /**
     * Computes partial results of the correlation or variance-covariance matrix algorithm
     * in the distributed processing mode
     * @return  Computed partial results of the correlation or variance-covariance matrix algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        partialResult = new PartialResult(getContext(), cObject, prec, method, ComputeMode.distributed,
                                          ComputeStep.step2Master);
        return partialResult;
    }

    /**
     * Computes the results of the correlation or variance-covariance matrix algorithm
     * in the distributed processing mode
     * @return  Computed results of the correlation or variance-covariance matrix algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        Result result = new Result(getContext(), cObject, prec, method, ComputeMode.distributed,
                                   ComputeStep.step2Master);
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix
     * algorithm and optionally tells the library to initialize the memory
     * @param partialResult         Structure to store partial results
     * @param initializationFlag    Flag that specifies whether partial results are initialized
     */
    public void setPartialResult(PartialResult partialResult, boolean initializationFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(),
                          initializationFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix algorithm
     * in the distributed processing mode
     * @param partialResult         Structure to store partial results
     */
    public void setPartialResult(PartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store the results of the correlation or variance-covariance matrix algorithm
     * in the distributed processing mode
     * @param result    Structure to store the results
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native void cSetPartialResult(long cAlgorithm, int prec, int method, long cPartialResult,
                                          boolean initializationFlag);

    /**
     * Releases the memory allocated for the native algorithm object
     */
    @Override
    public void dispose() {
        if (this.cDistributedIface != 0) {
            cDispose(this.cDistributedIface);
            this.cDistributedIface = 0;
        }
        super.dispose();
    }

    /**
     * Returns the newly allocated correlation or variance-covariance matrix algorithm
     * with a copy of input objects and parameters of this correlation or variance-covariance matrix algorithm
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract DistributedIface clone(DaalContext context);

    private native long cInitDistributedIface();
    private native void cDispose(long cDistributedIface);
}
