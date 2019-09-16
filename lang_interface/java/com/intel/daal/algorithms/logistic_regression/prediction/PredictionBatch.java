/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup logistic_regression_prediction_batch Batch
 * @ingroup logistic_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.classifier.Parameter;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Provides methods for logistic regression model-based prediction
 * <!-- \n<a href="DAAL-REF-LOGISTICREGRESSION-ALGORITHM">Logistic regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - Model class
 *      - PredictionInputId class
 *      - PredictionResultNumericTableId class
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    public PredictionInput  input;       /*!< %Input data */
    public PredictionMethod method;      /*!< %Prediction method for the algorithm */
    public Parameter        parameter;   /*!< Parameters of the algorithm */
    private Precision       prec;        /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a logistic regression prediction algorithm by copying
     * input objects and parameters of another logistic regression prediction algorithm
     * @param context   Context to manage logistic regression model-based prediction
     * @param other     %Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the logistic regression prediction algorithm in the batch processing mode
     * @param context   Context to manage logistic regression model-based prediction
     * @param cls       Data type to use in intermediate computations of logistic regression,
     *                  Double.class or Float.class
     * @param method    %Algorithm prediction method, @ref PredictionMethod
     * @param nClasses  Number of classes
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method, long nClasses) {
        super(context);

        this.method = method;
        if (this.method != PredictionMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nClasses);
        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes prediction results based on the model of the logistic regression algorithm
     * @return %Prediction results
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        return new PredictionResult(getContext(), this.cObject);
    }

    /**
     * Registers user-allocated memory to store the result of logistic regression model-based prediction
     * @param result Object to store the result of logistic regression model-based prediction
     */
    public void setResult(PredictionResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns a newly allocated logistic regression prediction algorithm
     * with a copy of the input objects of this logistic regression prediction algorithm
     * in the batch processing mode
     * @param context   Context to manage logistic regression model-based prediction
     *
     * @return Newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);
    private native long cInitParameter(long algAddr, int prec, int method);
    private native long cGetInput(long algAddr, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cGetResult(long algAddr, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
