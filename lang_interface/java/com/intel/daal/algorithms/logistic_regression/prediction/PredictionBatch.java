/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    public PredictionParameter        parameter;   /*!< Parameters of the algorithm */
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
        parameter = new PredictionParameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
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
        parameter = new PredictionParameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes prediction results based on the model of the logistic regression algorithm
     * @return %Prediction results
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        return new PredictionResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
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
