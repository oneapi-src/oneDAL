/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @defgroup bf_knn_classification_prediction_batch Batch
 * @ingroup bf_knn_classification_prediction
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the brute-force K nearest neighbors models
 */
package com.intel.daal.algorithms.bf_knn_classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.bf_knn_classification.Parameter;
import com.intel.daal.algorithms.bf_knn_classification.prediction.PredictionResult;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Runs brute-force k nearest neighbors model based prediction algorithm
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">brute-force k nearest neighbors algorithm description and usage models</a> -->
 *
 * @par References
 *      - PredictionMethod class
 *      - PredictioInput class
 *      - com.intel.daal.algorithms.classifier.prediction.NumericTableInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.ModelInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResultId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    public PredictionInput  input;     /*!< %Input data */
    public Parameter        parameter; /*!< Parameters of the algorithm */
    public PredictionMethod method;    /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs brute-force k nearest neighbors prediction algorithm by copying input objects and parameters
     * of another brute-force k nearest neighbors prediction algorithm
     * @param context   Context to manage the brute-force k nearest neighbors prediction algorithm
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
     * Constructs brute-force k nearest neighbors prediction algorithm
     * @param context   Context to manage the brute-force k nearest neighbors prediction algorithm
     * @param cls       Data type to use in intermediate computations of the brute-force k nearest neighbors prediction algorithm,
     *                  Double.class or Float.class
     * @param method    K nearest neighbors prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != PredictionMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
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
     * Returns the newly allocated brute-force k nearest neighbors prediction algorithm
     * with a copy of input objects and parameters of this brute-force k nearest neighbors prediction algorithm
     * @param context   Context to manage the brute-force k nearest neighbors prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    /**
     * Computes the result of brute-force k nearest neighbors model-based prediction
     * in the batch processing mode
     * @return Result of brute-force k nearest neighbors model-based prediction
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        PredictionResult result = new PredictionResult(getContext(), cObject);
        return result;
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
