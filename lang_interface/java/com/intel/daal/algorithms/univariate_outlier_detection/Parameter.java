/* file: Parameter.java */
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
 * @ingroup univariate_outlier_detection
 * @{
 */
package com.intel.daal.algorithms.univariate_outlier_detection;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__PARAMETER"></a>
 * @brief Parameters of the univariate outlier detection algorithm @DAAL_DEPRECATED
 */
@Deprecated
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
    }

    /**
     * Set initialization procedure for specifying initial parameters of the univariate outlier detection algorithm
     * @param initializationProcedure   Initialization procedure
     */
    public void setInitializationProcedure(InitializationProcedureIface initializationProcedure) {}

    /**
     * Gets the initialization procedure for setting the initial parameters of the univariate outlier detection algorithm
     * @return  Initialization procedure
     */
    public Object getInitializationProcedure() {
        return _initializationProcedure;
    }

    private Object _initializationProcedure;
}
/** @} */
