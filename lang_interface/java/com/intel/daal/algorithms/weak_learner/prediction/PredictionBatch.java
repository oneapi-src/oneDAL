/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup weak_learner_prediction Prediction
 * @brief Contains classes to make predictions based on the weak learner model
 * @ingroup weak_learner
 * @{
 */
/**
 * @defgroup weak_learner_prediction_batch Batch
 * @ingroup weak_learner_prediction
 * @{
 */
/**
 * @brief Contains classes for making predictions based on the weak learner model
 */
package com.intel.daal.algorithms.weak_learner.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHM__WEAK_LEARNER__PREDICTION__PREDICTIONBATCH"></a>
 *  @brief Base class for making predictions based on the weak learner model
 *
 *  @par References
 *      - com.intel.daal.algorithms.classifier.prediction.NumericTableInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.ModelInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResultId class
 *      - com.intel.daal.algorithms.weak_learner.Model class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionInput class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 *
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs algorithm for making predictions based on the weak learner model
     * by copying input objects and parameters of another algorithm
     * @param context   Context to manage the weak learner algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context, other);
    }

    /**
     * Constructs algorithm for making predictions based on the weak learner model
     * @param context   Context to manage the weak learner algorithm
     */
    public PredictionBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated algorithm for making predictions based on the weak learner model
     * with a copy of input objects and parameters of this algorithm
     * @param context   Context to manage the weak learner algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }
}
/** @} */
/** @} */
