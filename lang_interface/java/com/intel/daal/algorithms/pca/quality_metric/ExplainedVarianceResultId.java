/* file: ExplainedVarianceResultId.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup pca_quality_metric_explained_variance
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCERESULTTID"></a>
 * @brief Available identifiers of the result of explained variance quality metrics
 */
public final class ExplainedVarianceResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ExplainedVarianceResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int explainedVariancesId = 0;
    private static final int explainedVariancesRatiosId = 1;
    private static final int noiseVarianceId = 2;

    /*!< Explained variances */
    public static final ExplainedVarianceResultId explainedVariances       = new ExplainedVarianceResultId(explainedVariancesId);
    /*!< Explained variances ratios */
    public static final ExplainedVarianceResultId explainedVariancesRatios = new ExplainedVarianceResultId(explainedVariancesRatiosId);
    /*!< Noise variance */
    public static final ExplainedVarianceResultId noiseVariance            = new ExplainedVarianceResultId(noiseVarianceId);
}
/** @} */
