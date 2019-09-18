/* file: TrainingMethod.java */
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
 * @defgroup adaboost_training Training
 * @brief Contains classes for AdaBoost models training
 * @ingroup adaboost
 * @{
 */
package com.intel.daal.algorithms.adaboost.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training AdaBoost models
 */
public final class TrainingMethod {

    private int _value;

    /**
     * Constructs the training method object using the provided value
     * @param value     Value corresponding to the training method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training method object
     * @return Value corresponding to the training method object
     */
    public int getValue() {
        return _value;
    }

    private static final int Samme = 0;
    private static final int DefaultDense = Samme;
    private static final int SammeR = 1;

    /** Default AdaBoost training method */
    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense);
    /** SAMME algorithm */
    public static final TrainingMethod samme = new TrainingMethod(Samme);
    /** SAMME.R algorithm, need probabilities from weak learner prediction result */
    public static final TrainingMethod sammeR = new TrainingMethod(SammeR);
}
/** @} */
