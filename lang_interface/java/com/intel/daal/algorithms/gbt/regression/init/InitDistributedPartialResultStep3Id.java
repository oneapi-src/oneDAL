/* file: InitDistributedPartialResultStep3Id.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedPartialResultStep3Id"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the sixth step of the distributed processing mode
 */
public final class InitDistributedPartialResultStep3Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedPartialResultStep3Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3BinnedDataValue = 0;
    private static final int step3TransposedBinnedDataValue = 1;
    private static final int step3ResponseValue = 2;
    private static final int step3TreeOrderValue = 3;

    public static final InitDistributedPartialResultStep3Id step3BinnedData = new InitDistributedPartialResultStep3Id(step3BinnedDataValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3TransposedBinnedData = new InitDistributedPartialResultStep3Id(step3TransposedBinnedDataValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3Response = new InitDistributedPartialResultStep3Id(step3ResponseValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3TreeOrder = new InitDistributedPartialResultStep3Id(step3TreeOrderValue);
        /*!<  */
}
/** @} */
