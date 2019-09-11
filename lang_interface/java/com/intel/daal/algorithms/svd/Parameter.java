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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__PARAMETER"></a>
 * @brief Parameters of the compute() method of the SVD algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     *  Sets the format of the matrix of left singular vectors
     *  @param format  Format of the matrix of left singular vectors
     */
    public void setLeftSingularMatrixFormat(ResultFormat format) {
        cSetLeftSingularMatrixFormat(this.cObject, format.getValue());
    }

    /**
     *  Sets the format of the matrix of right singular vectors
     *  @param format  Format of the matrix of right singular vectors
     */
    public void setRightSingularMatrixFormat(ResultFormat format) {
        cSetRightSingularMatrixFormat(this.cObject, format.getValue());
    }

    /**
     *  Gets the format of the matrix of left singular vectors
     *  @return  Format of the matrix of left singular vectors
     */
    public ResultFormat getLeftSingularMatrixFormat() {
        ResultFormat format = ResultFormat.notRequired;
        int flag = cGetLeftSingularMatrixFormat(this.cObject);
        if (flag == ResultFormat.notRequired.getValue()) {
            format = ResultFormat.notRequired;
        } else {
            if (flag == ResultFormat.requiredInPackedForm.getValue()) {
                format = ResultFormat.requiredInPackedForm;
            }
        }

        return format;
    }

    /**
     *  Gets the format of the matrix of right singular vectors
     *  @return  Format of the matrix of right singular vectors
     */
    public ResultFormat getRightSingularMatrixFormat() {
        ResultFormat format = ResultFormat.notRequired;
        int flag = cGetRightSingularMatrixFormat(this.cObject);
        if (flag == ResultFormat.notRequired.getValue()) {
            format = ResultFormat.notRequired;
        } else {
            if (flag == ResultFormat.requiredInPackedForm.getValue()) {
                format = ResultFormat.requiredInPackedForm;
            }
        }

        return format;
    }

    private native long cParInit();

    private native void cSetLeftSingularMatrixFormat(long algAddr, int format);

    private native void cSetRightSingularMatrixFormat(long algAddr, int format);

    private native int cGetLeftSingularMatrixFormat(long algAddr);

    private native int cGetRightSingularMatrixFormat(long algAddr);
}
/** @} */
