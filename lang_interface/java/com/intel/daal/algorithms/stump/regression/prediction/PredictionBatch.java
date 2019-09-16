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
 * @defgroup stump_prediction_batch Batch
 * @ingroup stump_prediction
 * @{
 */
package com.intel.daal.algorithms.stump.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.stump.regression.Parameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts results of the decision stump regression
 *
 * @par References
 *      - Input class
 *      - Result class
 */
public class PredictionBatch extends com.intel.daal.algorithms.regression.prediction.PredictionBatch {
   public PredictionInput input;     /*!< %Input data */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */
    public Parameter parameter;

    /** @private */
    static {
        LibUtils.loadLibrary();
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
        this.prec      = other.prec;
        this.method    = other.method;
        this.cObject   = cClone(other.cObject, prec.getValue(), method.getValue());
        this.input     = new PredictionInput(getContext(), this.cObject, ComputeMode.batch);
        this.parameter = new Parameter( getContext(), cInitParameter(this.cObject,
                                                                     this.prec.getValue(),
                                                                     this.method.getValue()) );
    }

  /**
     * Constructs the decision stump training algorithm
     * @param context   Context to manage created stump training algorithm
     * @param cls       Data type to use in intermediate computations for the decision stump training algorithm,
     *                  Double.class or Float.class
     * @param method    the decision stump training method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);
        init(cls, method);
    }

    /**
     * Constructs the decision stump training algorithm
     * @param context   Context to manage created stump training algorithm
     */
    public PredictionBatch(DaalContext context) {
        super(context);
        init(Float.class, PredictionMethod.defaultDense);
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

    private void init(Class<? extends Number> cls, PredictionMethod method) {
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        final Precision prec = (cls == Double.class)
            ? Precision.doublePrecision
            : Precision.singlePrecision;
        init(prec, method);
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
