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
 * @defgroup brownboost_prediction_batch Batch
 * @ingroup brownboost_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on BrownBoost models
 */
package com.intel.daal.algorithms.brownboost.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.brownboost.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts BrownBoost classification results
 *
 * \par References
 *      - com.intel.daal.algorithms.brownboost.Model class
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    public PredictionInput  input;     /*!< %Input data */
    public PredictionMethod method;    /*!< %Prediction method for the algorithm */
    public Parameter        parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the BrownBoost prediction algorithm by copying input objects and parameters
     * of another BrownBoost prediction algorithm
     * @param context   Context to manage BrownBoost prediction
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
     * Constructs the BrownBoost prediction algorithm
     * @param context   Context to manage BrownBoost prediction
     * @param cls       Data type to use in intermediate computations for BrownBoost prediction,
     *                  Double.class or Float.class
     * @param method    BrownBoost prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);
        init(Precision.fromClass(cls), method);
    }

    /**
     * Returns the newly allocated BrownBoost prediction algorithm with a copy of input objects
     * and parameters of this BrownBoost prediction algorithm
     * @param context   Context to manage BrownBoost prediction
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private void init(Precision prec, PredictionMethod method) {
        this.prec      = prec;
        this.method    = method;
        this.cObject   = cInit(prec.getValue(), method.getValue());
        this.input     = new PredictionInput(getContext(), this.cObject, ComputeMode.batch);
        this.parameter = new Parameter( getContext(), cInitParameter(this.cObject,
                                                                     prec.getValue(),
                                                                     method.getValue()) );
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
