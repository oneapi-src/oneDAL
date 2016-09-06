/* file: ComputeMode.java */
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

package com.intel.daal.algorithms;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COMPUTEMODE"></a>
 * Computation modes of the Intel(R) Data Analytics Acceleration Library algorithms
 */
public final class ComputeMode {
    private int _value;

    public ComputeMode(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int batchValue       = 0;
    private static final int onlineValue      = 1;
    private static final int distributedValue = 2;

    /** Batch processing computation mode */
    public static final ComputeMode batch       = new ComputeMode(batchValue);
    /** Online mode - processing of data sets in blocks */
    public static final ComputeMode online      = new ComputeMode(onlineValue);
    /** Processing of data sets distributed across several devices */
    public static final ComputeMode distributed = new ComputeMode(distributedValue);
}
