/* file: PartialResultId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    private static final int PartialGoalFunction        = 2;
    private static final int PartialAssignments         = 3;
    private static final int PartialCandidatesDistances = 4;
    private static final int PartialCandidatesCentroids = 5;

    public static final PartialResultId nObservations              = new PartialResultId(
        NObservations);                                                       /*!< Number of assigned observations */
    public static final PartialResultId partialSums                = new PartialResultId(
        PartialSums);                                                         /*!< Sum of observations */
    public static final PartialResultId partialObjectiveFunction   = new PartialResultId(
        PartialObjectiveFunction);                                            /*!< Objective function value */
    public static final PartialResultId partialGoalFunction        = new PartialResultId(
        PartialGoalFunction);                                                 /*!< Objective function value @DAAL_DEPRECATED */
    public static final PartialResultId partialAssignments         = new PartialResultId(
        PartialAssignments);                                                  /*!< Assignments to clusters */
    public static final PartialResultId partialCandidatesDistances = new PartialResultId(
        PartialCandidatesDistances);                                          /*!< Objective function of observations most distant from their assigned cluster center */
    public static final PartialResultId partialCandidatesCentroids = new PartialResultId(
        PartialCandidatesCentroids);                                          /*!< observations most distant from their assigned cluster center */
}
/** @} */
