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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__RESULTTOCOMPUTEID"></a>
 * @brief Available identifiers of results of the classifier model training algorithm
 */
public final class ResultsToComputeId {

    public static final long computeClassLabels        = 0x0000000000000001L; /*!< Numeric table of size n x 1 with the predicted labels >*/
    public static final long computeClassProbabilities = 0x0000000000000002L; /*!< Numeric table of size n x p with the predicted class probabilities for each observation >*/
    public static final long computeClassLogProbabilities = 0x0000000000000004L; /*!< Numeric table of size n x p with the predicted class probabilities logarithm for each observation >*/
}
/** @} */
