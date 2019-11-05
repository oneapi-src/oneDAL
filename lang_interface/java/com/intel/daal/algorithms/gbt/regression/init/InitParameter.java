/* file: InitParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitParameter"></a>
 * @brief InitParameters of the DBSCAN computation method
 */
public class InitParameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public InitParameter(DaalContext context, long cInitParameter) {
        super(context);
        this.cObject = cInitParameter;
    }

    /**
     * Constructs a InitParameter
     * @param context               Context to manage the InitParameter of the DBSCAN algorithm
     */
    public InitParameter(DaalContext context) {
        super(context);

        long maxBins = 256;
        long minBinSize = 5;
        initialize(maxBins, minBinSize);
    }

    /**
     * Constructs a InitParameter
     * @param context               Context to manage the InitParameter of the DBSCAN algorithm
     * @param maxBins
     * @param minBinSize
     */
    public InitParameter(DaalContext context, long  maxBins, long minBinSize) {
        super(context);
        initialize(maxBins, minBinSize);
    }

    private void initialize(long  maxBins, long minBinSize) {
        setMaxBins(maxBins);
        setMinBinSize(minBinSize);
    }

    /**
     * Retrieves the radius of neighborhood
     * @return Radius of neighborhood
     */
    public long getMaxBins() {
        return cGetMaxBins(this.cObject);
    }

    /**
     * Retrieves the minimal number of observations in neighborhood of core observation
     * @return Minimal number of observations in neighborhood of core observation
     */
    public long getMinBinSize() {
        return cGetMinBinSize(this.cObject);
    }

    /**
    * Sets the radius of neighborhood
    * @param maxBins Radius of neighborhood
    */
    public void setMaxBins(long maxBins) {
        cSetMaxBins(this.cObject, maxBins);
    }

    /**
     * Sets the minimal number of observations in neighborhood of core observation
     * @param minBinSize Minimal number of observations in neighborhood of core observation
     */
    public void setMinBinSize(long minBinSize) {
        cSetMinBinSize(this.cObject, minBinSize);
    }

    private native long cGetMaxBins(long InitParameterAddress);
    private native long cGetMinBinSize(long InitParameterAddress);

    private native void cSetMaxBins(long InitParameterAddress, long maxBins);
    private native void cSetMinBinSize(long InitParameterAddress, long minBinSize);
}
/** @} */
