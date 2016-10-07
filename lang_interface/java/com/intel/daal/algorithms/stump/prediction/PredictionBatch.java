/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.stump.prediction;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.prediction.PredictionInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts results of the decision stump classification
 *
 * @par References
 *      - PredictionMethod class
 *      - Input class
 *      - Result class
 */
public class PredictionBatch extends com.intel.daal.algorithms.weak_learner.prediction.PredictionBatch {
    public PredictionInput      input;     /*!< %Input data */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the decision stump prediction algorithm by copying input objects
     * of another decision stump prediction algorithm
     * @param context   Context to manage the stump prediction algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new PredictionInput(getContext(), cObject, ComputeMode.batch);
    }

    /**
     * Constructs the decision stump prediction algorithm
     * @param context   Context to manage the stump prediction algorithm
     * @param cls       Data type to use in intermediate computations for the decision stump prediction algorithm,
     *                  Double.class or Float.class
     * @param method    the decision stump prediction method, @ref PredictionMethod
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
        input = new PredictionInput(getContext(), cObject, ComputeMode.batch);
    }

    /**
     * Returns the newly allocated decision stump prediction algorithm
     * with a copy of input objects and parameters of this decision stump prediction algorithm
     * @param context   Context to manage the stump prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
