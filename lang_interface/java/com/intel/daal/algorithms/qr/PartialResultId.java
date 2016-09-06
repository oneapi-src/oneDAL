/* file: PartialResultId.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.qr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the QR decomposition algorithm in the online processing mode and of the algorithm on the
 * first step in the distributed processing mode
 */
public final class PartialResultId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int outputOfStep1ForStep3Id = 0;
    private static final int outputOfStep1ForStep2Id = 1;

    /** DataCollection with data to be transfered to distributed step 3 */
    public static final PartialResultId outputOfStep1ForStep3 = new PartialResultId(outputOfStep1ForStep3Id);
    /** DataCollection with data to be transfered to distributed step 2 */
    public static final PartialResultId outputOfStep1ForStep2 = new PartialResultId(outputOfStep1ForStep2Id);
}
