/* file: PartialResultId.java */
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the SVD algorithm obtained in the online processing mode and in the first step in
 * the distributed processing mode
 */
public final class PartialResultId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public PartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int outputOfStep1ForStep3Id = 0;
    @Native private static final int outputOfStep1ForStep2Id = 1;

    /** DataCollection with data computed in the first step to be transferred to the third step in the distributed processing mode */
    public static final PartialResultId outputOfStep1ForStep3 = new PartialResultId(outputOfStep1ForStep3Id);
    /** DataCollection with data computed in the first step to be transferred to the second step in the distributed processing mode */
    public static final PartialResultId outputOfStep1ForStep2 = new PartialResultId(outputOfStep1ForStep2Id);
}
/** @} */
