/* file: Step4LocalCollectionInputId.java */
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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.dbscan;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP4LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection objects for the DBSCAN algorithm in the fourth step
 *        of the distributed processing mode
 */
public final class Step4LocalCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step4LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step4PartialSplitsValue = 2;
    private static final int step4PartialOrdersValue = 3;

    public static final Step4LocalCollectionInputId step4PartialSplits = new Step4LocalCollectionInputId(step4PartialSplitsValue);
       /*!< Collection of input tables containing information about split for current iteration of gemoetric repartitioning */
    public static final Step4LocalCollectionInputId step4PartialOrders = new Step4LocalCollectionInputId(step4PartialOrdersValue);
       /*!< Collection of input tables containing information about observations: identifier of initial block and index in initial block */
}
/** @} */
