/* file: TrainingBatch.java */
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
 * @defgroup ridge_regression_batch Batch
 * @ingroup ridge_regression_training
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ridge_regression.TrainParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__TRAININGBATCH"></a>
 * @brief Provides methods for ridge regression model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - Parameter class
 *      - Model class
 *      - ModelNormEq class
 *      - TrainingInputId class
 *      - TrainingResultId class
 */
public class TrainingBatch extends com.intel.daal.algorithms.TrainingBatch {
    public Input           input;     /*!< %Input data */
    public TrainParameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod  method;   /*!< %Training method for the algorithm */
    private Precision      prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a ridge regression training algorithm by copying input objects and parameters of another ridge regression training algorithm in the
     * batch processing mode
     * @param context   Context to manage ridge regression model-based training
     * @param other     %Algorithm to use as the source to initialize the input objects and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainParameter(getContext(), cInitTrainParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the ridge regression training algorithm in the batch processing mode
     * @param context   Context to manage ridge regression model-based training
     * @param cls       Data type to use in intermediate computations of ridge regression, Double.class or Float.class
     * @param method    %Algorithm computation method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;
        if (this.method != TrainingMethod.normEqDense) {
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
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainParameter(getContext(), cInitTrainParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of ridge regression model-based training in the batch processing mode
     * @return Result of ridge regression model-based training
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns a newly allocated ridge regression training algorithm with a copy of the input objects and parameters of this ridge regression
     * training algorithm in the batch processing mode
     * @param context   Context to manage ridge regression model-based training
     *
     * @return Newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitTrainParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);

}
/** @} */
