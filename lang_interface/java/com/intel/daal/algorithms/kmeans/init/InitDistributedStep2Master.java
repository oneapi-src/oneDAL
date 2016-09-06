/* file: InitDistributedStep2Master.java */
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

package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2MASTER"></a>
 * @brief Computes initial clusters for the K-Means algorithm in the distributed processing mode on the master node
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a>
 *
 * @par References
 *      - ComputeStep class. Computation step in the distributed processing mode
 *      - InitMethod class.  Methods of computing initial clusters for the algorithm
 *      - InitInputId class. Input objects for computing initial clusters for the algorithm
 *      - PartialResultId class.  Partial results of computing initial clusters for the algorithm
 *      - ResultId class. Results of computing initial clusters for the algorithm
 *      - InitDistributedStep2MasterInput class
 *      - Parameter class
 *      - InitPartialResult class
 *      - InitResult class
 *      - InitParameter class
 */
public class InitDistributedStep2Master extends AnalysisDistributed {
    public InitDistributedStep2MasterInput input;         /*!< %Input data */
    public InitParameter                   parameter;     /*!< Parameters for computing initial clusters */
    public InitMethod                      method;        /*!< Method for computing initial clusters */
    protected InitPartialResult            partialResult; /*!< Partial result of the initialization algorithm */
    private Precision                      prec;          /*!< Data type for computing initial clusters to use in intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs an algorithm for computing initial clusters for the K-Means algorithm in the second step
     * of the distributed processing mode by copying input objects and parameters of another algorithm
     * @param context     Context to manage initial clusters for the K-Means algorithm
     * @param other       An algorithm to be used as the source to initialize the input objects
     *                    and parameters of the algorithm
     */
    public InitDistributedStep2Master(DaalContext context, InitDistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        parameter = new InitParameter(getContext(), cInitParameter(cObject, prec.getValue(), this.method.getValue()), 0, 0);
        partialResult = null;
        input = new InitDistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs an algorithm for computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode
     * @param context   Context to manage initial clusters for the algorithm
     * @param cls       Data type to use in intermediate computations of initial clusters for the algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref InitMethod
     * @param nClusters Number of initial clusters for the K-Means algorithm
     */
    public InitDistributedStep2Master(DaalContext context, Class<? extends Number> cls, InitMethod method,
            long nClusters) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != InitMethod.defaultDense       &&
            this.method != InitMethod.deterministicDense && this.method != InitMethod.randomDense &&
            this.method != InitMethod.deterministicCSR   && this.method != InitMethod.randomCSR  ) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClusters);
        parameter = new InitParameter(getContext(), cInitParameter(cObject, prec.getValue(), this.method.getValue()), 0, 0);
        partialResult = null;
        input = new InitDistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes initial clusters for the K-Means algorithm in the second step of the distributed processing mode
     * @return  Partial results of computing initial clusters for the K-Means algorithm
     */
    @Override
    public InitPartialResult compute() {
        super.compute();
        partialResult = new InitPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes the results of K-Means initialization in the second step of the distributed processing mode
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
     * in the second step of the distributed processing mode
     * @param partialResult         Structure to store partial results of computing initial clusters for the K-Means algorithm
     * @param initFlag    Flag that specifies initialization of partial results
     */
    public void setPartialResult(InitPartialResult partialResult, boolean initFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(), initFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode
     * @param partialResult         Structure to store partial results of computing initial clusters for the K-Means algorithm
     */
    public void setPartialResult(InitPartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store the results of computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode
     * @param result    Structure to store the results of computing initial clusters for the K-Means algorithm
     */
    public void setResult(InitResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm for computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode with a copy of input objects and parameters of this algorithm
     * @param context     Context to manage initial clusters for the K-Means algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public InitDistributedStep2Master clone(DaalContext context) {
        return new InitDistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method, long nClusters);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult, boolean initFlag);

    private native long cGetResult(long cObject, int prec, int method);

    private native long cGetPartialResult(long cObject, int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
