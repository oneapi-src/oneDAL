/* file: TrainingBatch.java */
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
 * @defgroup weak_learner_training_batch Batch
 * @ingroup weak_learner_training
 * @{
 */
/**
 * @brief Contains classes for training the weak learner model
 */
package com.intel.daal.algorithms.weak_learner.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__TRAINING__TRAININGBATCH"></a>
 * @brief Base class for training the weak learner model in the batch processing mode
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 *      - TrainingResult class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs algorithm for training the weak learner model
     * by copying input objects and parameters of another algorithm
     * @param context   Context to manage the weak learner algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context, other);
    }

    /**
     * Constructs algorithm for training the weak learner model
     * @param context   Context to manage the weak learner algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Trains the weak learner model
     * @return Structure that contains computed training results
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated algorithm for training the weak learner model
     * with a copy of input objects and parameters of this algorithm
     * @param context   Context to manage the weak learner algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cGetResult(long algAddr);
}
/** @} */
