/* file: GroupOfBetasInputId.java */
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

package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASINPUTID"></a>
 * @brief Available identifiers of input objects for a single beta quality metrics
 */
public final class GroupOfBetasInputId {
    private int _value;

    public GroupOfBetasInputId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int ExpectedResponses   = 0;
    private static final int PredictedResponses = 1;
    private static final int PredictedReducedModelResponses = 2;

    /*!< Expected responses (Y), dependent variables */
    public static final GroupOfBetasInputId expectedResponses   = new GroupOfBetasInputId(ExpectedResponses);
    /*!< Predicted responses (Z) */
    public static final GroupOfBetasInputId predictedResponses = new GroupOfBetasInputId(PredictedResponses);
    /*!< Responses predicted by reduced model where p - p0 of p betas are set to zero */
    public static final GroupOfBetasInputId predictedReducedModelResponses = new GroupOfBetasInputId(PredictedReducedModelResponses);
}
