/* file: DistributedPartialResultStep11NumericTableId.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP11NUMERICTABLEID"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the eleventh step of the distributed processing mode
 */
public final class DistributedPartialResultStep11NumericTableId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep11NumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step11ClusterStructureValue = 0;
    private static final int step11FinishedFlagValue     = 1;

    public static final DistributedPartialResultStep11NumericTableId step11ClusterStructure =
        new DistributedPartialResultStep11NumericTableId(step11ClusterStructureValue);
        /*!< Table containing information about current clustering state of observations */
    public static final DistributedPartialResultStep11NumericTableId step11FinishedFlag =
        new DistributedPartialResultStep11NumericTableId(step11FinishedFlagValue);
        /*!< Table containing the flag indicating that the cluster numerating process is finished */
}
/** @} */
