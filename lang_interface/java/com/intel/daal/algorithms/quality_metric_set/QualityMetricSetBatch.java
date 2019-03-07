/* file: QualityMetricSetBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup quality_metric
 * @{
 */
/**
 * @brief Contains classes to compute a quality metric set
 */
package com.intel.daal.algorithms.quality_metric_set;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the quality metric algorithm
     * @param context   Context to manage the quality metric algorithm
     */
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
/** @} */
