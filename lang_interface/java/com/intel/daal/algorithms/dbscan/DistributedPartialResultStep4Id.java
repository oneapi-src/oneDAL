/* file: DistributedPartialResultStep4Id.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the fourth step of the distributed processing mode
 */
public final class DistributedPartialResultStep4Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep4Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partitionedDataValue          = 0;
    private static final int partitionedWeightsValue       = 1;
    private static final int partitionedPartialOrdersValue = 2;

    public static final DistributedPartialResultStep4Id partitionedData = new DistributedPartialResultStep4Id(partitionedDataValue);
        /*!< Collection of tables containing observations */
    public static final DistributedPartialResultStep4Id partitionedWeights = new DistributedPartialResultStep4Id(partitionedWeightsValue);
        /*!< Collection of tables containing weights of observations */
    public static final DistributedPartialResultStep4Id partitionedPartialOrders = new DistributedPartialResultStep4Id(partitionedPartialOrdersValue);
        /*!< Collection of tables containing information about observations: identifier of initial block and index in initial block */
}
/** @} */
