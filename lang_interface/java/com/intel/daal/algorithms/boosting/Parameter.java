/* file: Parameter.java */
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
