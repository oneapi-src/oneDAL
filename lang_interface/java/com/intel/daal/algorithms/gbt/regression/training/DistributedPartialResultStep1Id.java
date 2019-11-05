/* file: DistributedPartialResultStep1Id.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * @brief Available identifiers of partial results of the model-based training in the first step
 *        of the distributed processing mode
 */
public final class DistributedPartialResultStep1Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep1Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int responseValue = 0;
    private static final int optCoeffsValue = 1;
    private static final int treeOrderValue = 2;
    private static final int finalizedTreeValue = 3;
    private static final int step1TreeStructureValue = 4;

    public static final DistributedPartialResultStep1Id response = new DistributedPartialResultStep1Id(responseValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id optCoeffs = new DistributedPartialResultStep1Id(optCoeffsValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id treeOrder = new DistributedPartialResultStep1Id(treeOrderValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id finalizedTree = new DistributedPartialResultStep1Id(finalizedTreeValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id step1TreeStructure = new DistributedPartialResultStep1Id(step1TreeStructureValue);
        /*!<  */
}
/** @} */
