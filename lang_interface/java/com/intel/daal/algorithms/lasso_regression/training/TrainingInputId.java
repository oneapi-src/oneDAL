/* file: TrainingInputId.java */
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
 * @ingroup lasso_regression_training
 * @{
 */
package com.intel.daal.algorithms.lasso_regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__TRAINING__TRAININGINPUTID"></a>
 * @brief Available identifiers of input objects for lasso regression model-based training
 */
public final class TrainingInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the training input object identifier using the provided value
     * @param value     Value corresponding to the training input object identifier
     */
    public TrainingInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training input object identifier
     * @return Value corresponding to the training input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataId              = 0;
    private static final int dependentVariableId = 1;

    /** %Input data table */
    public static final TrainingInputId data              = new TrainingInputId(dataId);
    /** Values of the responses for the input data */
    public static final TrainingInputId dependentVariable = new TrainingInputId(dependentVariableId);

}
/** @} */
