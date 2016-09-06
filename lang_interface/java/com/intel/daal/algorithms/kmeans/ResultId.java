/* file: ResultId.java */
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

package com.intel.daal.algorithms.kmeans;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__RESULTID"></a>
 * @brief Available identifiers of the results of the K-Means algorithm
 */
public final class ResultId {
    private int _value;

    public ResultId(int value) {
        _value = value;
    }

     /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int centroidsValue    = 0;
    private static final int assignmentsValue  = 1;
    private static final int goalFunctionValue = 2;
    private static final int nIterationsValue  = 3;

    public static final ResultId centroids    = new ResultId(centroidsValue);    /*!< Centroids */
    public static final ResultId assignments  = new ResultId(assignmentsValue);  /*!< Assignment of observations to clusters */
    public static final ResultId goalFunction = new ResultId(goalFunctionValue); /*!< Value of the goal function */
    public static final ResultId nIterations  = new ResultId(nIterationsValue);  /*!< Number of executed iterations */

}
