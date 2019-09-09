/* file: InitPartialResultId.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm
 */
public final class InitPartialResultId {
    private int _value;

    /**
     * Constructs the initialization partial result object identifier using the provided value
     * @param value     Value corresponding to the initialization partial result object identifier
     */
    public InitPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization partial result object identifier
     * @return Value corresponding to the initialization partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialClusters       = 0;
    private static final int PartialCentroids      = 0;
    private static final int PartialClustersNumber = 1;

    /** Sum of observations */
    public static final InitPartialResultId partialCentroids      = new InitPartialResultId(PartialCentroids);
    /** Sum of observations @DAAL_DEPRECATED */
    public static final InitPartialResultId partialClusters       = new InitPartialResultId(PartialClusters);
    /** Number of assigned observations @DAAL_DEPRECATED */
    public static final InitPartialResultId partialClustersNumber = new InitPartialResultId(PartialClustersNumber);
}
/** @} */
