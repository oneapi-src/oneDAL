/* file: ResultsToComputeId.java */
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
 * @ingroup decision_forest
 * @{
 */
package com.intel.daal.algorithms.decision_forest;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__TRAINING__RESULTSTOCOMPUTEID"></a>
 * @brief Available computation flag identifiers for the decision forest result
 */
public final class ResultsToComputeId {

    public static final long computeOutOfBagError               = 0x0000000000000001L;/*!< Compute out-of-bag error */
    public static final long computeOutOfBagErrorPerObservation = 0x0000000000000002L;/*!< Compute out-of-bag error per observation */
}
/** @} */
