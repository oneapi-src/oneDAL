/* file: ResultsToComputeId.java */
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
 * @ingroup bf_knn_classification
 * @{
 */
package com.intel.daal.algorithms.bf_knn_classification;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__RESULTSTOCOMPUTEID"></a>
 * @brief Available identifiers to specify the result to compute
 */
public final class ResultsToComputeId {

    public static final long computeIndicesOfNeightbors = 0x0000000000000001L;
        /*!< The flag to compute indices of nearest neighbors */
    public static final long computeDistances           = 0x0000000000000002L;
        /*!< The flag to compute distances to nearest neighbors */
}
/** @} */
