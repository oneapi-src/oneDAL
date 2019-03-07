/* file: Parameter.java */
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARAMETER"></a>
 * @brief Parameters of the correlation or variance-covariance matrix algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public long cAlgorithm;

    /**
     * Constructs the parameter of the correlation or variance-covariance matrix algorithm
     * @param context   Context to manage the parameter of the correlation or variance-covariance matrix algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    public Parameter(DaalContext context, long cParameter, long cAlgorithm) {
        super(context);
        this.cObject = cParameter;
        this.cAlgorithm = cAlgorithm;
    }

    /**
     * Sets the parameter for the correlation or variance-covariance matrix algorithm
     * @param id    Identifier of the parameter, @ref OutputMatrixType
     */
    public void setOutputMatrixType(OutputMatrixType id) {
        cSetOutputDataType(this.cObject, id.getValue());
    }

    /**
     * Gets the parameter of the correlation or variance-covariance matrix algorithm
     * @return    Identifier of the parameter, @ref OutputMatrixType
     */
    public OutputMatrixType getOutputMatrixType() {
        OutputMatrixType id = new OutputMatrixType(cGetOutputDataType(this.cObject));
        return id;
    }

    public void setCParameter(long cParameter) {
        this.cObject = cParameter;
        cSetCParameterObject(this.cObject, this.cAlgorithm);
    }

    private native void cSetOutputDataType(long parAddr, int id);

    private native int cGetOutputDataType(long parAddr);

    private native void cSetCParameterObject(long parameterAddr, long algAddr);
}
/** @} */
