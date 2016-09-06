/* file: QualityMetricSetBatch.java */
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
 * @brief Contains classes to compute a quality metric set
 */
package com.intel.daal.algorithms.quality_metric_set;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__QUALITYMETRICSETBATCH"></a>
 * @brief Provides methods to compute a quality metric set of an algorithm in the batch processing mode
 */
public class QualityMetricSetBatch extends ContextClient {
    public long cObject;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public QualityMetricSetBatch(DaalContext context) {
        super(context);
    }

    /**
     * Computes results for a quality metric set in the batch processing mode
     * @return Structure that contains a computed quality metric set
     */
    public ResultCollection compute() {
        cCompute(this.cObject);
        return null;
    }

    /**
     * Releases memory allocated for the native object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native void cCompute(long algAddr);

    private native void cDispose(long algAddr);

}
