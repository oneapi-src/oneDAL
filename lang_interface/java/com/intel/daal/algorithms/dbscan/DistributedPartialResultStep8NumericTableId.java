/* file: DistributedPartialResultStep8NumericTableId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP8NUMERICTABLEID"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the eigth step of the distributed processing mode
 */
public final class DistributedPartialResultStep8NumericTableId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep8NumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step8ClusterStructureValue = 0;
    private static final int step8FinishedFlagValue     = 1;
    private static final int step8NClustersValue        = 2;

    public static final DistributedPartialResultStep8NumericTableId step8ClusterStructure =
        new DistributedPartialResultStep8NumericTableId(step8ClusterStructureValue);
        /*!< Table containing information about current clustering state of observations */
    public static final DistributedPartialResultStep8NumericTableId step8FinishedFlag =
        new DistributedPartialResultStep8NumericTableId(step8FinishedFlagValue);
        /*!< Table containing the flag indicating that the clustering process is finished */
    public static final DistributedPartialResultStep8NumericTableId step8NClusters =
        new DistributedPartialResultStep8NumericTableId(step8NClustersValue);
        /*!< Table containing the current number of clusters */
}
/** @} */
