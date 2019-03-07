/* file: QualityMetricResult.java */
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
package com.intel.daal.algorithms.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC__QUALITYMETRICRESULT"></a>
 * @brief  Base class for the result of quality metrics
 */
public class QualityMetricResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the quality metric algorithm
     * @param context   Context to manage the result of the quality metric algorithm
     */
    public QualityMetricResult(DaalContext context) {
        super(context);
        this.cObject = 0;
    }

    public QualityMetricResult(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }
}
/** @} */
