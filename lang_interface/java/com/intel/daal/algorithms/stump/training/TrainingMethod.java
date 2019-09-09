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
 * @defgroup stump Stump
 * @brief Contains classes to work with the decision stump training algorithm
 * @ingroup weak_learner
 * @{
 */
/**
 * @defgroup stump_training Training
 * @brief Contains classes to train the decision stump model
 * @ingroup stump
 * @{
 */
/**
 * @brief Contains classes to train the decision stump model
 */
package com.intel.daal.algorithms.stump.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods to train the decision stump model
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

    private static final int DefaultDense = 0;

    /** The default decision stump training method */
    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense);
}
/** @} */
/** @} */
