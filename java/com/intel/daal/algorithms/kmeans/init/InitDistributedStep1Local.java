/* file: InitDistributedStep1Local.java */
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
 * @defgroup kmeans_init_distributed Distributed
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP1LOCAL"></a>
 * @brief First step of computing initial clusters for the K-Means algorithm on local nodes
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
 *
 * @par References
 *      - ComputeStep class. Computation step in the distributed processing mode
 *      - InitInputId class. Input objects for computing initial clusters for the algorithm
 *      - PartialResultId class. Partial results of computing initial clusters for the algorithm
 *      - ResultId class. Results of computing initial clusters for the algorithm
 *      - InitDistributedStep1LocalInput class
 *      - Parameter class
 */
public class InitDistributedStep1Local extends AnalysisDistributed {
    public InitDistributedStep1LocalInput input;         /*!< %Input data */
    public InitParameter                  parameter;     /*!< Parameters for computing initial clusters */
    public InitMethod                     method;        /*!< %Method for computing initial clusters */
    protected InitPartialResult           partialResult; /*!< Partial result of the initialization algorithm */
    private Precision                     prec;          /*!< Data type for computing initial clusters to use in intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs an algorithm for computing initial clusters for the K-Means algorithm in the first step
     * of the distributed processing mode by copying input objects and parameters of another algorithm
     * @param context     Context to manage initial clusters for the K-Means algorithm
     * @param other       An algorithm to be used as the source to initialize the input objects
     *                    and parameters of the algorithm
     */
    public InitDistributedStep1Local(DaalContext context, InitDistributedStep1Local other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new InitDistributedStep1LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new InitParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()), 0, 0);
        partialResult = null;
    }

    /**
     * Constructs an algorithm for computing initial clusters for the K-Means algorithm in the first step
     * of the distributed processing mode
     * @param context     Context to manage initial clusters for the K-Means algorithm
     * @param cls         Data type to use in intermediate computations of initial clusters for the K-Means algorithm,
     *                    Double.class or Float.class
     * @param method      %Method of computing initial clusters for the algorithm, @ref InitMethod
     * @param nClusters   Number of initial clusters for the K-Means algorithm
     * @param nRowsTotal  Number of rows in all data sets in the distributed processing mode
     * @param offset      Offset in the total data set specifying the start of a block stored on a given local node
     */
    public InitDistributedStep1Local(DaalContext context, Class<? extends Number> cls, InitMethod method,
            long nClusters, long nRowsTotal, long offset) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != InitMethod.defaultDense       &&
            this.method != InitMethod.deterministicDense && this.method != InitMethod.randomDense &&
            this.method != InitMethod.plusPlusDense      && this.method != InitMethod.parallelPlusDense &&
            this.method != InitMethod.deterministicCSR   && this.method != InitMethod.randomCSR &&
            this.method != InitMethod.plusPlusCSR        && this.method != InitMethod.parallelPlusCSR
               ) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClusters, offset);

        input = new InitDistributedStep1LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));

        parameter = new InitParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()), 0, 0);
        parameter.setNClusters(nClusters);
        parameter.setNRowsTotal(nRowsTotal);
        parameter.setOffset(offset);
        partialResult = null;
    }

    /**
     * Computes initial clusters for the K-Means algorithm
     * @return  Partial results of computing initial clusters for the K-Means algorithm
     */
    @Override
    public InitPartialResult compute() {
        super.compute();
        partialResult = new InitPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes the results of K-Means initialization
     * @return  Results of K-Means initialization
     */
    @Override
    public InitResult finalizeCompute() {
        super.finalizeCompute();
        InitResult result = new InitResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of computing initial clusters for the K-Means algorithm
     * @param partialResult         Structure to store partial results of computing initial clusters for the K-Means algorithm
     * @param initFlag    Flag that specifies initialization of partial results
     */
    public void setPartialResult(InitPartialResult partialResult, boolean initFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(), initFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of computing initial clusters for the K-Means algorithm
     * @param partialResult         Structure to store partial results of computing initial clusters for the K-Means algorithm
     */
    public void setPartialResult(InitPartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store the results of computing initial clusters for the K-Means algorithm
     * @param result    Structure to store the results of computing initial clusters for the K-Means algorithm
     */
    public void setResult(InitResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm for computing initial clusters for the K-Means algorithm
     * in the first step of the distributed processing mode with a copy of input objects and parameters of this algorithm
     * @param context     Context to manage initial clusters for the K-Means algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public InitDistributedStep1Local clone(DaalContext context) {
        return new InitDistributedStep1Local(context, this);
    }

    private native long cInit(int prec, int method, long nClusters, long nShift);

    private native long cInitParameter(long addr, int prec, int method);

    private native long cGetInput(long addr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult, boolean initFlag);

    private native long cGetResult(long cObject, int prec, int method);

    private native long cGetPartialResult(long cObject, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
