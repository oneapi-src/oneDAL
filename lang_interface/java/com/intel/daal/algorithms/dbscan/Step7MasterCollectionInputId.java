/* file: Step7MasterCollectionInputId.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP7MASTERCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection objects for the DBSCAN algorithm in the seventh step
 *        of the distributed processing mode
 */
public final class Step7MasterCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step7MasterCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialFinishedFlagsValue = 0;

    public static final Step7MasterCollectionInputId partialFinishedFlags = new Step7MasterCollectionInputId(partialFinishedFlagsValue);
       /*!< Collection of input tables containing the flags indicating that the clustering process is finished */
}
/** @} */
