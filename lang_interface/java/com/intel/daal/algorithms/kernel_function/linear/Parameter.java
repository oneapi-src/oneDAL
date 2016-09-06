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

package com.intel.daal.algorithms.kernel_function.linear;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__PARAMETER"></a>
 * @brief Parameters for computing the linear kernel function k(X,Y) + b
 */
public class Parameter extends com.intel.daal.algorithms.kernel_function.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
    * Sets the linear kernel coefficient k
    * @param k   Linear kernel coefficient k
    */
    public void setK(double k) {
        cSetK(this.cObject, k);
    }

    /**
    * Gets the linear kernel coefficient k
    * @return  Linear kernel coefficient k
    */
    public double getK() {
        return cGetK(this.cObject);
    }

    /**
    * Sets the linear kernel coefficient b
    * @param b   Linear kernel coefficient b
    */
    public void setB(double b) {
        cSetB(this.cObject, b);
    }

    /**
    * Gets the linear kernel coefficient b
    * @return  Linear kernel coefficient b
    */
    public double getB() {
        return cGetB(this.cObject);
    }

    private native void cSetK(long parAddr, double k);
    private native void cSetB(long parAddr, double b);
    private native double cGetK(long parAddr);
    private native double cGetB(long parAddr);
}
