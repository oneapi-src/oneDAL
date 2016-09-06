/* file: DistributedStep3Local.java */
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

package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.TrainingDistributed;
import com.intel.daal.algorithms.implicit_als.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP3LOCAL"></a>
 * @brief Runs the implicit ALS training algorithm in the third step of the distributed processing mode
 */
public class DistributedStep3Local extends TrainingDistributed {
    public DistributedStep3LocalInput input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the implicit ALS training algorithm in the third step of the distributed processing mode
     * by copying input objects and parameters of another implicit ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep3Local(DaalContext context, DistributedStep3Local other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new DistributedStep3LocalInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the implicit ALS training algorithm in the third step of the distributed processing mode
     * @param context   Context to manage the implicit ALS training algorithm
     * @param cls       Data type to use in intermediate computations for the implicit ALS training algorithm,
     *                  Double.class or Float.class
     * @param method    Implicit ALS computation method, @ref TrainingMethod
     */
    public DistributedStep3Local(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;
        if (this.method != TrainingMethod.fastCSR && this.method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new DistributedStep3LocalInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes partial results of the implicit ALS training algorithm obtained in the third step of the distributed processing mode
     * @return Partial results of the implicit ALS training algorithm obtained in the third step of the distributed processing mode
     */
    @Override
    public DistributedPartialResultStep3 compute() {
        super.compute();
        return new DistributedPartialResultStep3(getContext(), cObject, prec, method);
    }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * obtained in the third step of the distributed processing mode
     * @param partialResult         Structure to store partial results of the implicit ALS training algorithm
     * obtained in the third step of the distributed processing mode
     */
    public void setPartialResult(DistributedPartialResultStep3 partialResult) {
        cSetPartialResult(this.cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated ALS training algorithm in the third step of the distributed processing mode
     * with a copy of input objects and parameters of this ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep3Local clone(DaalContext context) {
        return new DistributedStep3Local(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult);

    private native long cClone(long algAddr, int prec, int method);
}
