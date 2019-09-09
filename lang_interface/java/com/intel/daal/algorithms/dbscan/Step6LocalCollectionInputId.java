/* file: Step6LocalCollectionInputId.java */
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
