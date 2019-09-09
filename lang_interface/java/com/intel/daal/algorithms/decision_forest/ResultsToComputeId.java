/* file: ResultsToComputeId.java */
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
 * @ingroup decision_forest
 * @{
 */
package com.intel.daal.algorithms.decision_forest;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__TRAINING__RESULTSTOCOMPUTEID"></a>
 * @brief Available computation flag identifiers for the decision forest result
 */
public final class ResultsToComputeId {

    public static final long computeOutOfBagError               = 0x0000000000000001L;/*!< Compute out-of-bag error */
    public static final long computeOutOfBagErrorPerObservation = 0x0000000000000002L;/*!< Compute out-of-bag error per observation */
}
/** @} */
