/* file: TrainingBatch.java */
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
 * @defgroup boosting_training_batch Batch
 * @ingroup boosting_training
 * @{
 */
/**
 * @brief Contains classes for training models of %boosting classifiers
 */
package com.intel.daal.algorithms.boosting.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__TRAINING__TRAININGBATCH"></a>
 * @brief Base class for training models of %boosting algorithms in the batch processing mode
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResult class
 */
public abstract class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs boosting training algorithm
     * @param context   Context to manage boosting training algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated boosting training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage boosting training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingBatch clone(DaalContext context);
}
/** @} */
