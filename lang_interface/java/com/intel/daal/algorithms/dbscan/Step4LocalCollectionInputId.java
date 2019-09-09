/* file: Step4LocalCollectionInputId.java */
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
