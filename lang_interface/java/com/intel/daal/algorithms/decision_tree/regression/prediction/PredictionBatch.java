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
 * @defgroup decision_tree_regression_prediction_batch Batch
 * @ingroup decision_tree_regression_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on decision tree regression models
 */
package com.intel.daal.algorithms.decision_tree.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.regression.Parameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts decision tree regression regression results
 *
 * \par References
 *      - com.intel.daal.algorithms.decision_tree.regression.Model class
 *      - com.intel.daal.algorithms.decision_tree.regression.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.regression.prediction.PredictionBatch {
    protected Precision    prec;
    public PredictionInput input;   /*!< %Input data */
    public Parameter parameter;     /*!< Parameters of the algorithm */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the decision tree regression prediction algorithm by copying input objects and parameters
     * of another decision tree regression prediction algorithm
     * @param context   Context to manage decision tree regression prediction
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cGetInput(cObject, prec.getValue(), this.method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), this.method.getValue()));
    }

    /**
     * Constructs the decision tree regression prediction algorithm
     * @param context   Context to manage decision tree regression prediction
     * @param cls       Data type to use in intermediate computations for decision tree regression prediction,
     *                  Double.class or Float.class
     * @param method    decision tree regression prediction method, @ref PredictionMethod
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
        input = new PredictionInput(getContext(), cGetInput(cObject, prec.getValue(), this.method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), this.method.getValue()));
    }

    /**
     * Computes the results of decision tree regression prediction in the batch processing mode
     * @return Results of idecision tree regression prediction in the batch processing mode
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        PredictionResult result = new PredictionResult(getContext(), this.cObject);
        return result;
    }

    /**
     * Returns the newly allocated decision tree regression prediction algorithm with a copy of input objects
     * and parameters of this decision tree regression prediction algorithm
     * @param context   Context to manage decision tree regression prediction
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

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
