/* file: InitializationProcedure.java */
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
 * @ingroup multivariate_outlier_detection
 * @{
 */
package com.intel.daal.algorithms.multivariate_outlier_detection.defaultdense;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__DEFAULTDENSE__INITIALIZATIONPROCEDURE"></a>
 * @brief Class that specifies the default method for setting initial parameters of multivariate outlier detection algorithm \DAAL_DEPRECATED
 */
@Deprecated
public class InitializationProcedure extends InitializationProcedureIface {

    /**
     * Constructs the default initialization procedure
     * @param context   Context to manage the algorithm
     */
    public InitializationProcedure(DaalContext context) {
        super(context);
    }

    /**
     * Sets initial parameters of multivariate outlier detection algorithm
     * @param data        %Input data table of size n x p
     * @param location    Vector of mean estimates of size 1 x p
     * @param scatter     Measure of spread, the variance-covariance matrix of size p x p
     * @param threshold   Limit that defines the outlier region, the array of size 1 x 1 containing a non-negative number
     */
    @Override
    public void initialize(NumericTable data, NumericTable location, NumericTable scatter, NumericTable threshold) {}

    /**
    * Releases memory allocated for the native iface object
    */
    @Override
    public void dispose() {}
}
/** @} */
