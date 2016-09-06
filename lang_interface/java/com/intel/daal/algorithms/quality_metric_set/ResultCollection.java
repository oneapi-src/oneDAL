/* file: ResultCollection.java */
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

package com.intel.daal.algorithms.quality_metric_set;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Result;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 *  @brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 */
public class ResultCollection extends com.intel.daal.data_management.data.KeyValueDataCollection {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public ResultCollection(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public ResultCollection(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, (long)0);
        this.cObject = cInit(cAlgorithm);
    }

    /**
     * Adds an element with a key to the collection
     * @param key     Key value
     * @param value   %Result object
     */
    public void add(int key, Result value) {
        cAddResult(getCObject(), key, value.getCObject());
    }

    private native long cInit(long algaddr);

    private native void cAddResult(long collAddr, int key, long resAddr);

    protected native long cGetResult(long collAddr, int id);
}
