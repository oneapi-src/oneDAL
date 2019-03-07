/* file: PredictionResultsToComputeId.java */
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
 * @ingroup logistic_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PREDICTIONRESULTSTOCOMPUTEID"></a>
 * @brief Available identifiers of the result of logistic regression model-based prediction
 */
public final class PredictionResultsToComputeId {
    public static final long computeClassesLabels = 0x0000000000000001L;
    public static final long computeClassesProbabilities = 0x0000000000000002L;
    public static final long computeClassesLogProbabilities = 0x0000000000000004L;
}
/** @} */
