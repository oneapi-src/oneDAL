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
 * @defgroup adaboost_prediction_batch Batch
 * @ingroup adaboost_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on AdaBoost models
 */
package com.intel.daal.algorithms.adaboost.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.adaboost.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts AdaBoost classification results
 *
 * \par References
 *      - com.intel.daal.algorithms.adaboost.Model class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.boosting.prediction.PredictionBatch {
    public PredictionInput  input;     /*!< %Input data */
    public PredictionMethod method;    /*!< %Prediction method for the algorithm */
    public Parameter        parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the AdaBoost prediction algorithm by copying input objects and parameters
     * of another AdaBoost prediction algorithm
     * @param context   Context to manage AdaBoost prediction
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the AdaBoost prediction algorithm
     * @param context   Context to manage AdaBoost prediction
     * @param cls       Data type to use in intermediate computations for AdaBoost prediction,
     *                  Double.class or Float.class
     * @param method    AdaBoost prediction method, @ref PredictionMethod
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
        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated AdaBoost prediction algorithm with a copy of input objects
     * and parameters of this AdaBoost prediction algorithm
     * @param context   Context to manage AdaBoost prediction
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
