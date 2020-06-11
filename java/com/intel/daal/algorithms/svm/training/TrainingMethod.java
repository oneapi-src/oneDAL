/* file: TrainingMethod.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @defgroup svm_training Training
 * @brief Contains classes to train the SVM model
 * @ingroup svm
 * @{
 */
package com.intel.daal.algorithms.svm.training;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods to train the SVM model
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

    @Native private static final int boserValue   = 0;
    @Native private static final int thunderValue = 1;

    public static final TrainingMethod boser   = new TrainingMethod(boserValue);   /*!< Method proposed by Boser et al.   */
    public static final TrainingMethod thunder = new TrainingMethod(thunderValue); /*!< Method proposed by Thunder et al. */
}
/** @} */
