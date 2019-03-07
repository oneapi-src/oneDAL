/* file: InputCovariancesId.java */
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
 * @ingroup em_gmm_compute
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INPUTCOVARIANCESID"></a>
 * @brief Available identifiers of input covariance objects for the EM for GMM algorithm
 */
public final class InputCovariancesId {
    private int _value;

    /**
     * Constructs the input covariance object identifier using the provided value
     * @param value     Value corresponding to the input covariance object identifier
     */
    public InputCovariancesId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input covariance object identifier
     * @return Value corresponding to the input covariance object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int InputCovariancesId = 3;

    public static final InputCovariancesId inputCovariances = new InputCovariancesId(
            InputCovariancesId); /*!< %Collection of input covariances */
}
/** @} */
