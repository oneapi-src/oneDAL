/* file: DistributedStep2Master.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup low_order_moments_distributed
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDSTEP2MASTER"></a>
 * @brief Computes moments of low order in the distributed processing mode on the master node.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * @par References
 *      - ComputeStep class. Step of distributed processing
 *      - InputId class. Identifiers of input objects for the low order moments algorithm
 *      - PartialResultId class.  Partial results of the algorithm
 *      - ResultId class. Final results of the algorithm
 *      - Parameter class
 */
public class DistributedStep2Master extends AnalysisDistributed {
    public DistributedStep2MasterInput          input;     /*!< %Input data */
    public Method                               method;  /*!< Computation method for the algorithm */
    protected PartialResult            partialResult;     /*!< Partial result of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs low order moments algorithm by copying input objects
     * of another low order moments algorithm
     * @param context   Context to manage the low order moments algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep2Master(DaalContext context, DistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        partialResult = null;
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the low order moments algorithm
     * @param context   Context to manage the low order moments algorithm
     * @param cls       Data type to use in intermediate computations,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public DistributedStep2Master(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense && this.method != Method.singlePassDense
            && this.method != Method.sumDense && this.method != Method.fastCSR
            && this.method != Method.singlePassCSR && this.method != Method.sumCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        partialResult = null;
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes partial results of the low order moments algorithm
     * @return  Partial results of the algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        partialResult = new PartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes final results of the low order moments algorithm
     * @return  Final results of the algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of the low order moments algorithm and optionally tells
     * the library to initialize the memory
     * @param partialResult         Structure for storing partial results
     * @param initializationFlag    Flag that specifies whether the partial results are initialized
     */
    public void setPartialResult(PartialResult partialResult, boolean initializationFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(),
                          initializationFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of the low order moments algorithm
     * @param partialResult         Structure for storing partial results of the low order moments algorithm
     */
    public void setPartialResult(PartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store final results of the low order moments algorithm
     * @param result    Structure for storing final results of the low order moments algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated low order moments algorithm
     * with a copy of input objects of this algorithm
     * @param context   Context to manage the low order moments algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep2Master clone(DaalContext context) {
        return new DistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native void cSetPartialResult(long cAlgorithm, int prec, int method, long cPartialResult,
                                          boolean initializationFlag);

    private native long cGetResult(long cAlgorithm, int prec, int method);

    private native long cGetPartialResult(long cAlgorithm, int prec, int method);

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
