/* file: PartialResultId.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the K-Means algorithm
 */
public final class PartialResultId {
    private int _value;

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

    private static final int NObservations              = 0;
    private static final int PartialSums                = 1;
    private static final int PartialObjectiveFunction   = 2;
    private static final int PartialAssignments         = 3;
    private static final int PartialCandidatesDistances = 4;
    private static final int PartialCandidatesCentroids = 5;

    public static final PartialResultId nObservations              = new PartialResultId(
        NObservations);                                                       /*!< Number of assigned observations */
    public static final PartialResultId partialSums                = new PartialResultId(
        PartialSums);                                                         /*!< Sum of observations */
    public static final PartialResultId partialObjectiveFunction   = new PartialResultId(
        PartialObjectiveFunction);                                            /*!< Objective function value */
    public static final PartialResultId partialAssignments         = new PartialResultId(
        PartialAssignments);                                                  /*!< Assignments to clusters */
    public static final PartialResultId partialCandidatesDistances = new PartialResultId(
        PartialCandidatesDistances);                                          /*!< Objective function of observations most distant from their assigned cluster center */
    public static final PartialResultId partialCandidatesCentroids = new PartialResultId(
        PartialCandidatesCentroids);                                          /*!< observations most distant from their assigned cluster center */
}
/** @} */
