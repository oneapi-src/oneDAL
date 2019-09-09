/* file: Step10LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP10LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the tenth step
 *        of the distributed processing mode
 */
public final class Step10LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step10LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step10InputClusterStructureValue = 0;
    private static final int step10ClusterOffsetValue         = 1;

    public static final Step10LocalNumericTableInputId step10InputClusterStructure = new Step10LocalNumericTableInputId(step10InputClusterStructureValue);
       /*!< Input table containing information about current clustering state of observations */
    public static final Step10LocalNumericTableInputId step10ClusterOffset = new Step10LocalNumericTableInputId(step10ClusterOffsetValue);
       /*!< Input table containing the cluster numeration offset */
}
/** @} */
