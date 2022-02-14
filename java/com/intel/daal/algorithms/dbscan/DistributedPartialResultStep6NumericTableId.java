/* file: DistributedPartialResultStep6NumericTableId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP6NUMERICTABLEID"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the sixth step of the distributed processing mode
 */
public final class DistributedPartialResultStep6NumericTableId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep6NumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step6ClusterStructureValue = 0;
    private static final int step6FinishedFlagValue     = 1;
    private static final int step6NClustersValue        = 2;

    public static final DistributedPartialResultStep6NumericTableId step6ClusterStructure =
        new DistributedPartialResultStep6NumericTableId(step6ClusterStructureValue);
        /*!< Table containing information about current clustering state of observations */
    public static final DistributedPartialResultStep6NumericTableId step6FinishedFlag =
        new DistributedPartialResultStep6NumericTableId(step6FinishedFlagValue);
        /*!< Table containing the flag indicating that the clustering process is finished */
    public static final DistributedPartialResultStep6NumericTableId step6NClusters =
        new DistributedPartialResultStep6NumericTableId(step6NClustersValue);
        /*!< Table containing the current number of clusters */
}
/** @} */
