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
 * @defgroup kdtree_knn_classification_prediction_batch Batch
 * @ingroup kdtree_knn_classification_prediction
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the KD-tree based K nearest neighbors models
 */
package com.intel.daal.algorithms.kdtree_knn_classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.kdtree_knn_classification.prediction.PredictionResult;
import com.intel.daal.algorithms.kdtree_knn_classification.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Runs KD-tree based k nearest neighbors model based prediction algorithm
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">KD-tree based k nearest neighbors algorithm description and usage models</a> -->
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
     * Constructs KD-tree based k nearest neighbors prediction algorithm by copying input objects and parameters
     * of another KD-tree based k nearest neighbors prediction algorithm
     * @param context   Context to manage the KD-tree based k nearest neighbors prediction algorithm
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
     * Constructs KD-tree based k nearest neighbors prediction algorithm
     * @param context   Context to manage the KD-tree based k nearest neighbors prediction algorithm
     * @param cls       Data type to use in intermediate computations of the KD-tree based k nearest neighbors prediction algorithm,
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
     * Computes the result of KD-tree based k nearest neighbors prediction
     * in the batch processing mode
     * @return Result of KD-tree based k nearest neighbors prediction
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        PredictionResult result = new PredictionResult(getContext(), cObject);
        return result;
    }

    /**
     * Returns the newly allocated KD-tree based k nearest neighbors prediction algorithm
     * with a copy of input objects and parameters of this KD-tree based k nearest neighbors prediction algorithm
     * @param context   Context to manage the k nearest neighbors prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
