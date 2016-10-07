/* file: InitParameter.java */
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

package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARAMETER"></a>
 * @brief Parameters of the implicit ALS algorithm
 */
public class InitParameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitParameter(DaalContext context, long parAddr) {
        super(context);
        this.cObject = parAddr;
    }

    /**
     * Sets the nUsers parameter
     * @param fullNUsers
     */
    public void setFullNUsers(long fullNUsers) {
        cSetFullNUsers(this.cObject, fullNUsers);
    }

    /**
     * Gets the value of the nUsers parameter
     * @return nUsers
     */
    public long getFullNUsers() {
        return cGetFullNUsers(this.cObject);
    }

    /**
     * Sets the nFactors parameter
     * @param nFactors
     */
    public void setNFactors(long nFactors) {
        cSetNFactors(this.cObject, nFactors);
    }

    /**
     * Gets the value of the nFactors parameter
     * @return nFactors
     */
    public long getNFactors() {
        return cGetNFactors(this.cObject);
    }

    /**
    * Sets the seed parameter
    * @param seed
    */
    public void setSeed(long seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * Gets the value of the seed parameter
     * @return seed
     */
    public long getSeed() {
        return cGetSeed(this.cObject);
    }

    private native void cSetFullNUsers(long algAddr, long nUsers);

    private native long cGetFullNUsers(long algAddr);

    private native void cSetNFactors(long algAddr, long nFactors);

    private native long cGetNFactors(long algAddr);

    private native void cSetSeed(long algAddr, long seed);

    private native long cGetSeed(long algAddr);
}
