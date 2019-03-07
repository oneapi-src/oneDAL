/* file: Parameter.java */
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
 * @ingroup boosting
 */
/**
 * @brief Contains base classes for working with %boosting classifiers
 */
package com.intel.daal.algorithms.boosting;

import com.intel.daal.algorithms.weak_learner.prediction.PredictionBatch;
import com.intel.daal.algorithms.weak_learner.training.TrainingBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__PARAMETER"></a>
 * @brief Base class for the parameters of the %boosting algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the algorithm for weak learner model training
     * @param weakLearnerTraining    %Algorithm for weak learner model training
     */
    public void setWeakLearnerTraining(TrainingBatch weakLearnerTraining) {
        cSetWlTraining(this.cObject, weakLearnerTraining.cObject);
    }

    /**
     * Retrieves the algorithm for weak learner model training
     * @return %Algorithm for weak learner model training
     */
    public TrainingBatch getWeakLearnerTraining() {
        TrainingBatch weakLearnerTraining = new TrainingBatch(getContext());
        weakLearnerTraining.cObject = cGetWlTraining(this.cObject);
        return weakLearnerTraining;
    }

    /**
     * Sets the algorithm for prediction based on a weak learner model
     * @param weakLearnerPrediction  %Algorithm for prediction based on a weak learner model
     */
    public void setWeakLearnerPrediction(PredictionBatch weakLearnerPrediction) {
        cSetWlPrediction(this.cObject, weakLearnerPrediction.cObject);
    }

    /**
     * Retrieves the algorithm for prediction based on a weak learner model
     * @return %Algorithm for prediction based on a weak learner model
     */
    public PredictionBatch getWeakLearnerPrediction() {
        PredictionBatch weakLearnerPrediction = new PredictionBatch(getContext());
        weakLearnerPrediction.cObject = cGetWlPrediction(this.cObject);
        return weakLearnerPrediction;
    }

    private native void cSetWlTraining(long parAddr, long algAddr);

    private native long cGetWlTraining(long parAddr);

    private native void cSetWlPrediction(long parAddr, long algAddr);

    private native long cGetWlPrediction(long parAddr);

}
/** @} */
