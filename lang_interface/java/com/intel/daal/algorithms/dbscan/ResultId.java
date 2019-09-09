/* file: ResultId.java */
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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.dbscan;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__RESULTID"></a>
 * @brief Available identifiers of the results of the DBSCAN algorithm
 */
public final class ResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int assignmentsValue      = 0;
    private static final int nClustersValue        = 1;
    private static final int coreIndicesValue      = 2;
    private static final int coreObservationsValue = 3;

    public static final ResultId assignments      = new ResultId(assignmentsValue);         /*!< Assignments of observations to clusters */
    public static final ResultId nClusters        = new ResultId(nClustersValue);           /*!< Number of clusters */
    public static final ResultId coreIndices      = new ResultId(coreIndicesValue);         /*!< Indices of core observations */
    public static final ResultId coreObservations = new ResultId(coreObservationsValue);    /*!< Core observations */

}
/** @} */
