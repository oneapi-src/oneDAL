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
 * @ingroup kdtree_knn_classification
 * @{
 */
package com.intel.daal.algorithms.kdtree_knn_classification;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__RESULTSTOCOMPUTEID"></a>
 * @brief Available identifiers to specify the result to compute
 */
public final class ResultsToComputeId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private long _value;

    /**
     * Constructs the input data usage mode object identifier using the provided value
     * @param value     Value corresponding to the input data usage mode object identifier
     */
    public ResultsToComputeId(long value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input data usage mode object identifier
     * @return Value corresponding to the input data usage mode object identifier
     */
    public long getValue() {
        return _value;
    }

    public static final long computeIndicesOfNeightborsId = 0x0000000000000001L;
    public static final long computeDistancesId           = 0x0000000000000002L;

    public static final ResultsToComputeId computeIndicesOfNeightbors = new ResultsToComputeId(computeIndicesOfNeightborsId);
        /*!< The flag to compute indices of nearest neighbors */
    public static final ResultsToComputeId computeDistances           = new ResultsToComputeId(computeDistancesId);
        /*!< The flag to compute distances to nearest neighbors */
}
/** @} */
