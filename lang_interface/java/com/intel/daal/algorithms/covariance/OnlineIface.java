/* file: OnlineIface.java */
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

package com.intel.daal.algorithms.covariance;

import com.intel.daal.algorithms.AnalysisOnline;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINEIFACE"></a>
 * @brief %Base interface for the correlation or variance-covariance matrix algorithm in the online processing mode
 * \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation or variance-covariance matrix algorithm description and usage models</a>
 *
 * @par References
 *      - Method class.  Computation methods
 *      - InputId class. Identifiers of input objects
 *      - PartialResultId class. Identifiers of partial results
 *      - ResultId class. Identifiers of the results
 *      - Input class
 *      - OnlineParameter class
 *      - Result class
 */
public abstract class OnlineIface extends AnalysisOnline {
    public long cOnlineIface; /*!< Pointer to the inner implementation of the service callback functionality */

    public Input          input;     /*!< %Input data */
    public OnlineParameter  parameter;  /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    public Precision        prec; /*!< Precision of computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__COVARIANCE__ONLINEIFACE__ONLINEIFACE"></a>
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     */
    public OnlineIface(DaalContext context) {
        super(context);
        this.cOnlineIface = cInitOnlineIface();
    }

    /**
     * Releases the memory allocated for the native algorithm object
     */
    @Override
    public void dispose() {
        if (this.cOnlineIface != 0) {
            cDispose(this.cOnlineIface);
            this.cOnlineIface = 0;
        }
        super.dispose();
    }

    /**
     * Computes partial results of the correlation or variance-covariance matrix algorithm in the online processing mode
     * @return  Computed partial results of the correlation or variance-covariance matrix algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        PartialResult partialResult = new PartialResult(getContext(), cObject, prec, method, ComputeMode.online);
        return partialResult;
    }

    /**
     * Computes the results of the correlation or variance-covariance matrix algorithm in the online processing mode
     * @return  Computed results of the correlation or variance-covariance matrix algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        Result result = new Result(getContext(), cObject, prec, method, ComputeMode.online);
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix
     * algorithm and optionally tells the library to initialize the memory
     * @param partialResult         Structure to store partial results
     * @param initializationFlag    Flag that specifies whether partial results are initialized
     */
    public void setPartialResult(PartialResult partialResult, boolean initializationFlag) {
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(), initializationFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     * @param partialResult         Structure to store partial results
     */
    public void setPartialResult(PartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store the results of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     * @param result    Structure to store the results of the correlation or variance-covariance matrix algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated correlation or variance-covariance matrix algorithm in the online processing mode
     * with a copy of input objects and parameters of this correlation or variance-covariance matrix algorithm
     * @param context   Context to manage the correlation or variance-covariance matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract OnlineIface clone(DaalContext context);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native void cSetPartialResult(long cAlgorithm, int prec, int method, long cPartialResult,
                                          boolean initializationFlag);

    protected native long cInitOnlineIface();

    private native void cDispose(long cOnlineIface);
}
