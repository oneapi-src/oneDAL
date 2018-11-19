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
 * @ingroup kernel_function
 * @{
 */
/**
 * @brief Contains classes for computing kernel functions
 */
package com.intel.daal.algorithms.kernel_function;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__PARAMETER"></a>
 * @brief Optional parameters for computing kernel functions
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
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
/** @} */
