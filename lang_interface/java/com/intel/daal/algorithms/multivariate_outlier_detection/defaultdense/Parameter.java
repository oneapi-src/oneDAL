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
 * @ingroup multivariate_outlier_detection_defaultdense
 * @{
 */
package com.intel.daal.algorithms.multivariate_outlier_detection.defaultdense;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__DEFAULTDENSE__PARAMETER"></a>
 * @brief Parameters for the multivariate outlier detection compute() used with the defaultDense method @DAAL_DEPRECATED
 */
@Deprecated
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);

        this.cObject = cParameter;
        _initializationProcedure = null;
    }

    /**
     * Set initialization procedure for specifying initial parameters of the multivariate outlier detection algorithm
     * @param initializationProcedure   Initialization procedure
     */
    public void setInitializationProcedure(InitializationProcedureIface initializationProcedure) {}

    /**
     * Get initialization procedure for specifying initial parameters of the multivariate outlier detection algorithm
     * @return  Initialization procedure
     */
    public InitializationProcedureIface getInitializationProcedure() {
        return _initializationProcedure;
    }

    private InitializationProcedureIface _initializationProcedure;
}
/** @} */
