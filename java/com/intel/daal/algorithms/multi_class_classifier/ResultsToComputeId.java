/* file: ResultsToComputeId.java */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__RESULTTOCOMPUTEID"></a>
 * @brief Available identifiers of results of the multi-class_classifier model training algorithm
 */
public final class ResultsToComputeId {

    public static final long computeClassLabels      = 0x0000000000000001L; /*!< Numeric table of size n x 1 with the predicted labels >*/
    public static final long computeDecisionFunction = 0x0000000000000032L; /*!< Numeric table of size n x (k*(k-1)/2) with the decision function for each observation >*/
}
/** @} */
