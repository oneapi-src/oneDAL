/* file: InitializationProcedureIface.java */
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
 * @ingroup multivariate_outlier_detection
 * @{
 */
package com.intel.daal.algorithms.multivariate_outlier_detection.defaultdense;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INITIALIZATIONPROCEDUREIFACE"></a>
 * @brief Abstract interface class for setting initial parameters of multivariate outlier detection algorithm \DAAL_DEPRECATED
 */
@Deprecated
public abstract class InitializationProcedureIface extends SerializableBase {

    /**
     * Constructs the initialization procedure iface
     * @param context   Context to manage the algorithm
     */
    public InitializationProcedureIface(DaalContext context) {
        super(context);
    }

    /**
     * Sets initial parameters of multivariate outlier detection algorithm
     * @param data        %Input data table of size n x p
     * @param location    Vector of mean estimates of size 1 x p
     * @param scatter     Measure of spread, the variance-covariance matrix of size p x p
     * @param threshold   Limit that defines the outlier region, the array of size 1 x 1 containing a non-negative number
     */
    abstract public void initialize(NumericTable data, NumericTable location, NumericTable scatter,
                                    NumericTable threshold);
}
/** @} */
