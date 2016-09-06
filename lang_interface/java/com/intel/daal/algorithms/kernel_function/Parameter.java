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

/**
 * @brief Contains classes for computing kernel functions
 */
package com.intel.daal.algorithms.kernel_function;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__PARAMETER"></a>
 * @brief Optional parameters for computing kernel functions
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the default parameter of the kernel function
     * @param context      Context to manage the parameter
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
    * Sets the mode of computing kernel functions
    * @param computationMode   Computation mode
    */
    public void setComputationMode(ComputationMode computationMode) {
        cSetComputationMode(this.cObject, computationMode.getValue());
    }

    /**
    * Sets the index of the vector in the set X
    * @param rowIndexX    Index of the vector in the set X
    */
    public void setRowIndexX(long rowIndexX) {
        cSetRowIndexX(this.cObject, rowIndexX);
    }

    /**
    * Gets the index of the vector in the set X
    * @return  index of the vector in the set X
    */
    public long getRowIndexX() {
        return cGetRowIndexX(this.cObject);
    }

    /**
    * Sets the index of the vector in the set Y
    * @param rowIndexY    Index of the vector in the set Y
    */
    public void setRowIndexY(long rowIndexY) {
        cSetRowIndexY(this.cObject, rowIndexY);
    }

    /**
    * Gets the index of the vector in the set Y
    * @return  index of the vector in the set Y
    */
    public long getRowIndexY() {
        return cGetRowIndexY(this.cObject);
    }

    /**
    * Sets the index of the result of the kernel function computation
    * @param rowIndexResult    Index of the result of the kernel function computation
    */
    public void setRowIndexResult(long rowIndexResult) {
        cSetRowIndexResult(this.cObject, rowIndexResult);
    }

    /**
    * Gets the index of the result of the kernel function computation
    * @return  index of the result of the kernel function computation
    */
    public long getRowIndexResult() {
        return cGetRowIndexResult(this.cObject);
    }

    private native void cSetComputationMode(long parAddr, int computationModeId);
    private native void cSetRowIndexX(long parAddr, long rowIndexX);
    private native void cSetRowIndexY(long parAddr, long rowIndexY);
    private native void cSetRowIndexResult(long parAddr, long rowIndexResult);
    private native long cGetRowIndexX(long parAddr);
    private native long cGetRowIndexY(long parAddr);
    private native long cGetRowIndexResult(long parAddr);
}
