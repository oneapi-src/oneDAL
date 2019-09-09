/* file: DistributedStep3LocalInputId.java */
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3LOCALINPUTID"></a>
 * @brief Partial results from previous steps of the SVD algorithm in the distributed processing mode, required by the third step
 */
public final class DistributedStep3LocalInputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the local node input object identifier using the provided value
     * @param value     Value corresponding to the local node input object identifier
     */
    public DistributedStep3LocalInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the local node input object identifier
     * @return Value corresponding to the local node input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int inputOfStep3FromStep1Id = 0;
    @Native private static final int inputOfStep3FromStep2Id = 1;

    /** DataCollection with data transferred from the first step to the third step in the distributed processing mode */
    public static final DistributedStep3LocalInputId inputOfStep3FromStep1 = new DistributedStep3LocalInputId(
            inputOfStep3FromStep1Id);
    /** DataCollection with data transferred from the second step to the third step in the distributed processing mode */
    public static final DistributedStep3LocalInputId inputOfStep3FromStep2 = new DistributedStep3LocalInputId(
            inputOfStep3FromStep2Id);
}
/** @} */
