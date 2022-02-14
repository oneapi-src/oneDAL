/* file: DistributedStep2MasterInputId.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup covariance_distributed
 * @{
 */
package com.intel.daal.algorithms.covariance;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * @brief Available identifiers of master-node input objects for the correlation or variance-covariance matrix algorithm
 */
public final class DistributedStep2MasterInputId {
    private int _value;

    /**
     * Constructs the master-node input object identifier using the provided value
     * @param value     Value corresponding to the master-node input object identifier
     */
    public DistributedStep2MasterInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the master-node input object identifier
     * @return Value corresponding to the master-node input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialResultsValue = 0;

    /** Collection of partial results obtained on local nodes */
    public static final DistributedStep2MasterInputId partialResults = new DistributedStep2MasterInputId(
        partialResultsValue);
}
/** @} */
