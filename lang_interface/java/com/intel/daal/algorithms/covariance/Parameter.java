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

package com.intel.daal.algorithms.covariance;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARAMETER"></a>
 * @brief Parameters of the correlation or variance-covariance matrix algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public long cAlgorithm;

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
