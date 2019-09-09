/* file: ResultNumericTableId.java */
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
 * @ingroup decision_forest_classification_training
 * @{
 */
package com.intel.daal.algorithms.decision_forest.classification.training;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING_RESULTNUMERICTABLEID"></a>
 * @brief Available identifiers of the result of decision forest model-based training
 */
public final class ResultNumericTableId {
    private int _value;

    /**
     * Constructs the result numeric table object identifier using the provided value
     * @param value     Value corresponding to the result numeric table object identifier
     */
    public ResultNumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result numeric table object identifier
     * @return Value corresponding to the result numeric table object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int outOfBagErrorId = 1;
    @Native private static final int variableImportanceId = 2;
    @Native private static final int outOfBagErrorPerObservationId = 3;

    public static final ResultNumericTableId outOfBagError = new ResultNumericTableId(outOfBagErrorId);
                        /*!< %Out-of-bag error result */
    public static final ResultNumericTableId variableImportance = new ResultNumericTableId(variableImportanceId);
                        /*!< %Variable importance result */
    public static final ResultNumericTableId outOfBagErrorPerObservation = new ResultNumericTableId(outOfBagErrorPerObservationId);
                        /*!< %Out-of-bag error perf observation result */
}
/** @} */
