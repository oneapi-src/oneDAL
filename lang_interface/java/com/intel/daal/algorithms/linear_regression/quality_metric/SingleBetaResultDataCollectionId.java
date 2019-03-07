/* file: SingleBetaResultDataCollectionId.java */
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
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETARESULTDATACOLLECTIONID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class SingleBetaResultDataCollectionId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public SingleBetaResultDataCollectionId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int BetaCovariances   = 5;

    /*!< Numeric tables with nBeta x nBeta variance-covariance matrix for betas of each response (dependent variable) */
    public static final SingleBetaResultDataCollectionId betaCovariances   = new SingleBetaResultDataCollectionId(BetaCovariances);
}
/** @} */
