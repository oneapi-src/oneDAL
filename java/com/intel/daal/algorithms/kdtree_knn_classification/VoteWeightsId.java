/* file: VoteWeightsId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__VOTEWEIGHTSID"></a>
 * @brief Weight function used in prediction voting
 */
public final class VoteWeightsId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input data usage mode object identifier using the provided value
     * @param value     Value corresponding to the input data usage mode object identifier
     */
    public VoteWeightsId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input data usage mode object identifier
     * @return Value corresponding to the input data usage mode object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int voteUniformId  = 0;
    private static final int voteDistanceId = 1;

    public static final VoteWeightsId voteUniform  = new VoteWeightsId(voteUniformId);
        /*!< Uniform weights for neighbors for prediction voting. All neighbors are weighted equally */
    public static final VoteWeightsId voteDistance = new VoteWeightsId(voteDistanceId);
        /*!< Weight neighbors by the inverse of their distance. Closer neighbors of a query point will have a
             greater influence than neighbors that are further away */
}
/** @} */
