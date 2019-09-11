/* file: Step6LocalCollectionInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP6LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection objects for the DBSCAN algorithm in the sixth step
 *        of the distributed processing mode
 */
public final class Step6LocalCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step6LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int haloDataValue        = 2;
    private static final int haloDataIndicesValue = 3;
    private static final int haloWeightsValue     = 4;
    private static final int haloBlocksValue      = 5;

    public static final Step6LocalCollectionInputId haloData = new Step6LocalCollectionInputId(haloDataValue);
       /*!< Collection of input tables containing halo observations */
    public static final Step6LocalCollectionInputId haloDataIndices = new Step6LocalCollectionInputId(haloDataIndicesValue);
       /*!< Collection of input tables containing indices of halo observations */
    public static final Step6LocalCollectionInputId haloWeights = new Step6LocalCollectionInputId(haloWeightsValue);
       /*!< Collection of input tables containing weights of halo observations */
    public static final Step6LocalCollectionInputId haloBlocks = new Step6LocalCollectionInputId(haloBlocksValue);
       /*!< Collection of input tables containing identifiers of blocks for halo observations */
}
/** @} */
