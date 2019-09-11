/* file: DistributedStep2MasterPartialResult.java */
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
 * @ingroup qr_distributed
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP2MASTERPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of  the QR decomposition algorithm on the second
 * step in the distributed processing mode
 */
public final class DistributedStep2MasterPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns partial result of the QR decomposition algorithm.
     * @param id    Identifier of partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(DistributedPartialResultCollectionId id) {
        if (id == DistributedPartialResultCollectionId.outputOfStep2ForStep3) {
            return new KeyValueDataCollection(getContext(), cGetKeyValueDataCollection(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of the QR decomposition algorithm with the matrix R calculated
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public Result get(DistributedPartialResultId id) {
        if (id == DistributedPartialResultId.finalResultFromStep2Master) {
            return new Result(getContext(), cGetResult(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetKeyValueDataCollection(long presAddr, int id);

    private native long cGetResult(long presAddr, int id);
}
/** @} */
