/* file: CovarianceStorageId.java */
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
 * @ingroup em_gmm
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__COVARIANCESTORAGEID"></a>
 * @brief Available identifiers of covariance types in the EM for GMM algorithm
 */
public final class CovarianceStorageId {
    private int _value;

    /**
     * Constructs the covariance type object identifier using the provided value
     * @param value     Value corresponding to the covariance type object identifier
     */
    public CovarianceStorageId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the covariance type object identifier
     * @return Value corresponding to the covariance type object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int fullValue      = 0;
    private static final int diagonalValue  = 1;

    public static final CovarianceStorageId full     = new CovarianceStorageId(fullValue);      /*!< Full */
    public static final CovarianceStorageId diagonal = new CovarianceStorageId(diagonalValue);  /*!< Diagonal */
}
/** @} */
