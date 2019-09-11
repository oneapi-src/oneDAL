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
 * @defgroup stump_prediction_batch Batch
 * @ingroup stump_prediction
 * @{
 */
package com.intel.daal.algorithms.stump.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Predicts results of the decision stump classification
 *
 * @par References
 *      - Input class
 *      - Result class
 */
public class PredictionBatch extends com.intel.daal.algorithms.weak_learner.prediction.PredictionBatch {
    public PredictionInput      input;     /*!< %Input data */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

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
/** @} */
