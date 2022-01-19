/* file: TrainingDistributedStep2Master.java */
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
 * @ingroup linear_regression_distributed
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.TrainingDistributed;
import com.intel.daal.algorithms.linear_regression.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__TRAININGDISTRIBUTEDSTEP2MASTER"></a>
 * @brief Runs linear regression model-based training in the second step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-LINEARREGRESSION-ALGORITHM">Linear regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - Model class
 *      - ModelNormEq class
 *      - ModelQR class
 *      - MasterInputId class
 *      - PartialResultId class
 *      - TrainingResultId class
 */
public class TrainingDistributedStep2Master extends TrainingDistributed {
    public DistributedStep2MasterInput input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a linear regression training algorithm by copying input objects
     * and parameters of another linear regression training algorithm
     * @param context   Context to manage linear regression model-based training
     * @param other     %Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingDistributedStep2Master(DaalContext context, TrainingDistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the linear regression training algorithm in the second step
     * of the distributed processing mode
     * @param context   Context to manage linear regression model-based training
     * @param cls       Data type to use in intermediate computations of linear regression,
     *                  Double.class or Float.class
     * @param method    %Algorithm computation method, @ref TrainingMethod
     */
    public TrainingDistributedStep2Master(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;
        if (this.method != TrainingMethod.normEqDense && this.method != TrainingMethod.qrDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes a partial result of linear regression model-based training in the second step of the distributed processing mode
     * @return Partial result of linear regression model-based training in the second step of the distributed processing mode
     */
    @Override
    public PartialResult compute() {
        super.compute();
        PartialResult presult = new PartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return presult;
    }

    /**
     * Computes the result of linear regression model-based training in the second step of the distributed processing mode
     * @return Result of linear regression model-based training in the second step of the distributed processing mode
     */
    @Override
    public TrainingResult finalizeCompute() {
        super.finalizeCompute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns a newly allocated linear regression training algorithm
     * with a copy of the input objects and parameters of this linear
     * regression training algorithm in the second step of the distributed processing mode
     * @param context   Context to manage linear regression model-based training
     *
     * @return Newly allocated algorithm
     */
    @Override
    public TrainingDistributedStep2Master clone(DaalContext context) {
        return new TrainingDistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cGetPartialResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
