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
 * @defgroup decision_tree_classification_prediction_batch Batch
 * @ingroup decision_tree_classification_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on decision tree classification models
 */
package com.intel.daal.algorithms.decision_tree.classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.decision_tree.classification.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts decision tree classification classification results
 *
 * \par References
 *      - com.intel.daal.algorithms.decision_tree.classification.Model class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    public PredictionInput      input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the decision tree classification prediction algorithm by copying input objects and parameters
     * of another decision tree classification prediction algorithm
     * @param context   Context to manage decision tree classification prediction
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        method = other.method;
        prec = other.prec;

        cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new PredictionInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the decision tree classification prediction algorithm
     * @param context   Context to manage decision tree classification prediction
     * @param cls       Data type to use in intermediate computations for decision tree classification prediction,
     *                  Double.class or Float.class
     * @param method    decision tree classification prediction method, @ref PredictionMethod
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

        cObject = cInit(prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cGetInput(cObject, prec.getValue(), this.method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(cObject, prec.getValue(), this.method.getValue()));
    }

    /**
     * Returns the newly allocated decision tree classification prediction algorithm with a copy of input objects
     * and parameters of this decision tree classification prediction algorithm
     * @param context   Context to manage decision tree classification prediction
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
