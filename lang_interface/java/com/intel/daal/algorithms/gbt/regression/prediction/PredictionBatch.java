/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup gbt_regression_prediction_batch Batch
 * @ingroup gbt_regression_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on gradient boosted trees regression models
 */
package com.intel.daal.algorithms.gbt.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.gbt.regression.prediction.Parameter;
import com.intel.daal.algorithms.gbt.regression.prediction.PredictionInput;
import com.intel.daal.algorithms.gbt.regression.prediction.PredictionResult;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts gradient boosted trees regression regression results
 *
 * \par References
 *      - com.intel.daal.algorithms.gbt.regression.Model class
 *      - com.intel.daal.algorithms.gbt.regression.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.Prediction {
    protected Precision    prec;
    public PredictionInput input;   /*!< %Input data */
    public Parameter parameter;     /*!< Parameters of the algorithm */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the gradient boosted trees regression prediction algorithm by copying input objects and parameters
     * of another gradient boosted trees regression prediction algorithm
     * @param context   Context to manage gradient boosted trees regression prediction
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Constructs the gradient boosted trees regression prediction algorithm
     * @param context   Context to manage gradient boosted trees regression prediction
     * @param cls       Data type to use in intermediate computations for gradient boosted trees regression prediction,
     *                  Double.class or Float.class
     * @param method    gradient boosted trees regression prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
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
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Computes the results of gradient boosted trees regression prediction in the batch processing mode
     * @return Results of igradient boosted trees regression prediction in the batch processing mode
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        PredictionResult result = new PredictionResult(getContext(), cObject, prec, method);
        return result;
    }

    /**
     * Returns the newly allocated gradient boosted trees regression prediction algorithm with a copy of input objects
     * and parameters of this gradient boosted trees regression prediction algorithm
     * @param context   Context to manage gradient boosted trees regression prediction
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method, int cmode);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
