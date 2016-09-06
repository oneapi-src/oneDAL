/* file: InitPartialResultId.java */
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

package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm
 */
public final class InitPartialResultId {
    private int _value;

    public InitPartialResultId(int value) {
        _value = value;
    }

     /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialClustersNumber = 0;
    private static final int PartialClusters       = 1;

    /** Number of assigned observations */
    public static final InitPartialResultId partialClustersNumber = new InitPartialResultId(PartialClustersNumber);
    /** Sum of observations */
    public static final InitPartialResultId partialClusters       = new InitPartialResultId(PartialClusters);
}
