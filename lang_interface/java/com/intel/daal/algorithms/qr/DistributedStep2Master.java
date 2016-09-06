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

package com.intel.daal.algorithms.qr;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP2MASTER"></a>
 * @brief Computes the results of the QR decomposition algorithm on the second step in the distributed processing mode
 * \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a>
 *
 * @par References
 *      - Method class.  Computation methods for the algorithm
 *      - DistributedStep2MasterInputId class. Identifiers of  input objects for the QR decomposition algorithm
 *      - DistributedPartialResultId class. Identifiers of partial results of the QR decomposition algorithm
 *      - DistributedPartialResultCollectionId class. Identifiers of partial results of the QR decomposition algorithm
 *      - ResultId class. Identifiers of the results of the QR decomposition algorithm
 *      - DistributedStep2MasterInput class
 *      - DistributedStep2MasterPartialResult class
 *      - Result class
 */
public class DistributedStep2Master extends AnalysisDistributed {
    public DistributedStep2MasterInput          input;     /*!< %Input data */
    public Method                               method;  /*!< Computation method for the algorithm */
    private DistributedStep2MasterPartialResult presult;   /*!< Partial result of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep2Master(DaalContext context, DistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param cls       Data type to use in intermediate computations for the QR decomposition algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public DistributedStep2Master(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = InitDistributed(prec.getValue(), method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the QR decomposition algorithm
     * @return  Partial results of the second step of the QR decomposition algorithm in the distributed processing mode
     */
    @Override
    public DistributedStep2MasterPartialResult compute() {
        super.compute();
        presult = new DistributedStep2MasterPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return presult;
    }

    /**
     * Computes final results of the QR decomposition algorithm
     * @return  Final results of the QR decomposition algorithm
     */
    @Override
    public Result finalizeCompute() {
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep2Master clone(DaalContext context) {
        return new DistributedStep2Master(context, this);
    }

    private native long InitDistributed(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cGetPartialResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
