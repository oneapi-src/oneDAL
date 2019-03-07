/* file: DistributedStep2MasterInputId.java */
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
 * @ingroup low_order_moments_distributed
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * @brief Available identifiers of master-node input objects for the low order momemnts algorithm
 */
public final class DistributedStep2MasterInputId {
    private int _value;

    /**
     * Constructs the master input object identifier using the provided value
     * @param value     Value corresponding to the master input object identifier
     */
    public DistributedStep2MasterInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the master input object identifier
     * @return Value corresponding to the master input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialResultsValue = 0;

    /** Collection of partial results obtained on local nodes */
    public static final DistributedStep2MasterInputId partialResults = new DistributedStep2MasterInputId(
            partialResultsValue);
}
/** @} */
