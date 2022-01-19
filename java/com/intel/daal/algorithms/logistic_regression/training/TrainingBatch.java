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
 * @defgroup logistic_regression_batch Batch
 * @ingroup logistic_regression_training
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.logistic_regression.training.TrainingParameter;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__TRAININGBATCH"></a>
 * @brief Provides methods for logistic regression model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTICREGRESSION-ALGORITHM">Logistic regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - TrainingParameter class
 *      - Model class
 *      - TrainingResult class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public TrainingParameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod  method;   /*!< %Training method for the algorithm */
    private Precision      prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a logistic regression training algorithm by copying input objects and parameters of another logistic regression training algorithm in the
     * batch processing mode
     * @param context   Context to manage logistic regression model-based training
     * @param other     %Algorithm to use as the source to initialize the input objects and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new TrainingParameter(getContext(), cInitTrainParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the logistic regression training algorithm in the batch processing mode
     * @param context   Context to manage logistic regression model-based training
     * @param cls       Data type to use in intermediate computations of logistic regression, Double.class or Float.class
     * @param method    %Algorithm computation method, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method, long nClasses) {
        super(context);

        this.method = method;

        if (method != TrainingMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nClasses);
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new TrainingParameter(getContext(), cInitTrainParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of logistic regression model-based training in the batch processing mode
     * @return Result of logistic regression model-based training
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns a newly allocated logistic regression training algorithm with a copy of the input objects and parameters of this logistic regression
     * training algorithm in the batch processing mode
     * @param context   Context to manage logistic regression model-based training
     *
     * @return Newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitTrainParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);

}
/** @} */
