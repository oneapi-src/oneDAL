/* file: DistributedPartialResultStep11NumericTableId.java */
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
