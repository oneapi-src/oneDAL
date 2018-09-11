/* file: ExplainedVarianceInputId.java */
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
 * @defgroup pca_quality_metric_explained_variance Explained Variance Coefficient
 * @ingroup pca_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCEINPUTID"></a>
 * @brief Available identifiers of input objects for a explained variance quality metrics
 */
public final class ExplainedVarianceInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public ExplainedVarianceInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int eigenValuesId = 0;

    /*!< Eigenvalues of PCA */
    public static final ExplainedVarianceInputId eigenValues = new ExplainedVarianceInputId(eigenValuesId);
}
/** @} */
