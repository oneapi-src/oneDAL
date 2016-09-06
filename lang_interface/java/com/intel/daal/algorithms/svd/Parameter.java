/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.svd;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__PARAMETER"></a>
 * @brief Parameters of the compute() method of the SVD algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
