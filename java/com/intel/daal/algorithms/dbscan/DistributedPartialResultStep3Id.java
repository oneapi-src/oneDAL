/* file: DistributedPartialResultStep3Id.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the third step of the distributed processing mode
 */
public final class DistributedPartialResultStep3Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep3Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int splitValue = 0;

    public static final DistributedPartialResultStep3Id split = new DistributedPartialResultStep3Id(splitValue);
        /*!< Table containing information about split for current iteration of geometric repartitioning */
}
/** @} */
