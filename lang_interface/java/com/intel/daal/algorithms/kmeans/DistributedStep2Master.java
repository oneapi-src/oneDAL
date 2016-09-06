/* file: DistributedStep2Master.java */
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

package com.intel.daal.algorithms.kmeans;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDSTEP2MASTER"></a>
 * @brief Computes K-Means in the distributed processing mode on the master node
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a>
 *
 * @par References
 *      - ComputeStep class. Step of the algorithm in the distributed processing mode
 *      - Method class.  Computation methods of the algorithm
 *      - InputId class. Input objects for the algorithm
 *      - PartialResultId class.  Partial results of the algorithm
 *      - DistributedStep2MasterInput class
 *      - Parameter class
 *      - PartialResult class
 *      - Result class
 */
public class DistributedStep2Master extends AnalysisDistributed {
    public DistributedStep2MasterInput input;         /*!< %Input data */
    public Method                      method;        /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the K-Means algorithm by copying input objects and parameters
     * of another K-Means algorithm
     * @param context   Context to manage algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep2Master(DaalContext context, DistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the K-Means algorithm
     * @param context   Context to manage algorithm
     * @param cls       Data type to use in intermediate computations for the algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method of the algorithm, @ref Method
     * @param nClusters Number of clusters for the algorithm
     */
    public DistributedStep2Master(DaalContext context, Class<? extends Number> cls, Method method, long nClusters) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense && this.method != Method.lloydCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClusters);

        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the K-Means algorithm
     * @return  Partial results of the K-Means algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        return new PartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the results of the K-Means algorithm
     * @return  Results of the K-Means algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the K-Means algorithm
     * @param partialResult         Structure to store partial results of the K-Means algorithm
     */
    public void setPartialResult(PartialResult partialResult) {
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * @param result    Structure to store the results of the K-Means  algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated K-Means algorithm with a copy of input objects
     * and parameters of this K-Means algorithm
     * @param context   Context to manage algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep2Master clone(DaalContext context) {
        return new DistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method, long nClusters);

    private native long cGetInput(long addr, int prec, int method);

    private native long cGetResult(long addr, int prec, int method);

    private native void cSetResult(long addr, int prec, int method, long cResult);

    private native long cGetPartialResult(long addr, int prec, int method);

    private native void cSetPartialResult(long addr, int prec, int method, long cPartialResult);

    private native long cClone(long addr, int prec, int method);
}
