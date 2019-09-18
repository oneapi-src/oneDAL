/* file: Model.java */
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
 * @defgroup stump_regression regression
 * @brief Contains classes for decision stump regression algorithm
 * @ingroup stump
 * @{
 */

package com.intel.daal.algorithms.stump.regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__MODEL"></a>
 * @brief %Model of the regression trained by decision stump regression algorithm in batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.regression.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     *  Returns the split feature
     *  @return Index of the feature over which the split is made
     */
    public long getSplitFeature() {
        return cGetSplitFeature(this.cObject);
    }

    /**
     *  Returns a value of the feature that defines the split
     *  @return Value of the feature over which the split is made
     */
    public double getSplitValue() {
        return cGetSplitValue(this.cObject);
    }

    /**
     *  Returns an average of the weighted responses for the "left" subset
     *  @return Average of the weighted responses for the "left" subset
     */
    public double getLeftValue() {
        return cGetLeftValue(this.cObject);
    }

    /**
     *  Returns an average of the weighted responses for the "right" subset
     *  @return Average of the weighted responses for the "right" subset
     */
    public double getRightValue() {
        return cGetRightValue(this.cObject);
    }

    private native long cGetSplitFeature(long selfPtr);
    private native double cGetSplitValue(long selfPtr);
    private native double cGetLeftValue(long selfPtr);
    private native double cGetRightValue(long selfPtr);
}
/** @} */
