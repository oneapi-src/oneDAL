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

package com.intel.daal.algorithms.kernel_function.rbf;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RBF__PARAMETER"></a>
 * \brief Parameters for computing the radial base function (RBF) kernel
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
    * Sets the RBF kernel coefficient
    * @param sigma   RBF kernel coefficient
    */
    public void setSigma(double sigma) {
        cSetSigma(this.cObject, sigma);
    }

    /**
    * Gets the RBF kernel coefficient
    * @return  RBF kernel coefficient
    */
    public double getSigma() {
        return cGetSigma(this.cObject);
    }

    private native void cSetSigma(long parAddr, double sigma);
    private native double cGetSigma(long parAddr);
}
