/* file: ComputeStep.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COMPUTESTEP"></a>
 * Describes on which node the computation stage is done in distributed computations
 */
public final class ComputeStep {
    private int _value;

    /**
     * Constructs the compute step object using the provided value
     * @param value     Value corresponding to the compute step object
     */
    public ComputeStep(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the compute step object
     * @return Value corresponding to the compute step object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int step1LocalValue  = 0;
    @Native private static final int step2MasterValue = 1;
    @Native private static final int step3LocalValue  = 2;

    /** Processing is done on local nodes */
    public static final ComputeStep step1Local  = new ComputeStep(step1LocalValue);
    /** Processing is done on master nodes */
    public static final ComputeStep step2Master = new ComputeStep(step2MasterValue);
    /** Finalization is done on local nodes */
    public static final ComputeStep step3Local  = new ComputeStep(step3LocalValue);
}
/** @} */
