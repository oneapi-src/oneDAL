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
 * @defgroup binary_adaboost_prediction_batch Batch
 * @ingroup binary_adaboost_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on binary_adaboost models
 */
package com.intel.daal.algorithms.binary_adaboost.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.binary_adaboost.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__binary_adaboost__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts binary_adaboost classification results
 *
 * \par References
 *      - com.intel.daal.algorithms.binary_adaboost.Model class
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
     * Constructs the binary_adaboost prediction algorithm by copying input objects and parameters
     * of another binary_adaboost prediction algorithm
     * @param context   Context to manage binary_adaboost prediction
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
     * Constructs the binary_adaboost prediction algorithm
     * @param context   Context to manage binary_adaboost prediction
     * @param cls       Data type to use in intermediate computations for binary_adaboost prediction,
     *                  Double.class or Float.class
     * @param method    binary_adaboost prediction method, @ref PredictionMethod
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
     * Returns the newly allocated binary_adaboost prediction algorithm with a copy of input objects
     * and parameters of this binary_adaboost prediction algorithm
     * @param context   Context to manage binary_adaboost prediction
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
