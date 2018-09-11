/* file: OutputMatrixType.java */
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__OUTPUTMATRIXTYPE"></a>
 * @brief Available types of the computed correlation or variance-covariance matrix
 */
public final class OutputMatrixType {
    private int _value;

    /**
     * Constructs the computed correlation or variance-covariance matrix object using the provided value
     * @param value     Value corresponding to the computed correlation or variance-covariance matrix object
     */
    public OutputMatrixType(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the computed correlation or variance-covariance matrix object
     * @return Value corresponding to the computed correlation or variance-covariance matrix object
     */
    public int getValue() {
        return _value;
    }

    private static final int CovarianceMatrix  = 0;
    private static final int CorrelationMatrix = 1;

    public static final OutputMatrixType covarianceMatrix  = new OutputMatrixType(
            CovarianceMatrix);  /*!< Variance-Covariance matrix */
    public static final OutputMatrixType correlationMatrix = new OutputMatrixType(
            CorrelationMatrix);  /*!< Correlation matrix */
}
/** @} */
