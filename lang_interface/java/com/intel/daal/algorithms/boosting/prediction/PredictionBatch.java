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
 * @defgroup boosting_prediction Prediction
 * @brief Contains classes for prediction based on boosting models
 * @ingroup boosting
 * @{
 */
/**
 * @defgroup boosting_prediction_batch Batch
 * @ingroup boosting_prediction
 * @{
 */
/**
 * @brief Contains classes for predictions based on %boosting classifiers models
 */
package com.intel.daal.algorithms.boosting.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Base class for training models of %boosting algorithms in the batch processing mode
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.prediction.NumericTableInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.ModelInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResultId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionInput class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 */
public abstract class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs boosting prediction algorithm
     * @param context   Context to manage boosting prediction algorithm
     */
    public PredictionBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated boosting prediction algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage boosting prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract PredictionBatch clone(DaalContext context);
}
/** @} */
/** @} */
