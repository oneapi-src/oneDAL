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
 * @defgroup logitboost_training Training
 * @brief Contains classes for LogitBoost models training
 * @ingroup logitboost
 * @{
 */
package com.intel.daal.algorithms.logitboost.training;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training LogitBoost models
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

    @Native private static final int Friedman = 0;
    private static final int DefaultDense = Friedman;

    /** Default method proposed by Friedman et al. */
    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense);
    public static final TrainingMethod friedman = new TrainingMethod(Friedman);
}
/** @} */
