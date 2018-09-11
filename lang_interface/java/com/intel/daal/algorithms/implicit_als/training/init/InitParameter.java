/* file: InitParameter.java */
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
 * @ingroup implicit_als_init
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARAMETER"></a>
 * @brief Parameters of the implicit ALS initialization algorithm
 */
public class InitParameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
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
    * @DAAL_DEPRECATED
    * Sets the seed parameter
    * @param seed
    */
    public void setSeed(long seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
    * @DAAL_DEPRECATED
     * Gets the value of the seed parameter
     * @return seed
     */
    public long getSeed() {
        return cGetSeed(this.cObject);
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native void cSetFullNUsers(long algAddr, long nUsers);

    private native long cGetFullNUsers(long algAddr);

    private native void cSetNFactors(long algAddr, long nFactors);

    private native long cGetNFactors(long algAddr);

    private native void cSetSeed(long algAddr, long seed);

    private native long cGetSeed(long algAddr);

    private native void cSetEngine(long cObject, long cEngineObject);
}
/** @} */
