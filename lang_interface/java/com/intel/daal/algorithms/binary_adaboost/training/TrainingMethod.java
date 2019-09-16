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
 * @defgroup binary_adaboost_training Training
 * @brief Contains classes for binary_adaboost models training
 * @ingroup binary_adaboost
 * @{
 */
package com.intel.daal.algorithms.binary_adaboost.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__binary_adaboost__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training binary_adaboost models
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

    /** Default binary_adaboost training method */
    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense);
}
/** @} */
