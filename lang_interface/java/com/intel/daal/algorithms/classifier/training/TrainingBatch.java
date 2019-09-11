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
 * @defgroup classifier_training_batch Batch
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.Result;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__TRAININGBATCH"></a>
 * @brief Algorithm class for training the classifier model
 *
 * @par References
 *      - InputId class
 *      - TrainingResultId class
 *      - TrainingInput class
 *      - TrainingResult class
 */
public abstract class TrainingBatch extends com.intel.daal.algorithms.TrainingBatch {
    public TrainingInput input;
    protected Precision  prec;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the base classifier training algorithm by copying input objects and parameters
     * of another base classifier training algorithm
     * @param context   Context to manage the classifier training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        input = other.input;
        prec = other.prec;
    }

    /**
     * Constructs the base classifier training algorithm
     * @param context   Context to manage the classifier training algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Computes results of the classifier training algorithm
     * \return Results of the classifier training algorithm
     */
    @Override
    public Result compute() {
        super.compute();
        return null;
    }

    /**
     * Returns the newly allocated base classifier training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage the classifier training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingBatch clone(DaalContext context);
}
/** @} */
