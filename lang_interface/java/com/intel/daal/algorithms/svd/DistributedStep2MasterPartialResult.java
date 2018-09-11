/* file: DistributedStep2MasterPartialResult.java */
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP2MASTERPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm in the second step in the
 * distributed processing mode
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
     * Returns partial results of the SVD algorithm.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm parameter
     * @param id    Identifier of the parameter
     * @return      Parameter that corresponds to the given identifier
     */
    public KeyValueDataCollection get(DistributedPartialResultCollectionId id) {
        if (id == DistributedPartialResultCollectionId.outputOfStep2ForStep3) {
            return new KeyValueDataCollection(getContext(), cGetKeyValueDataCollection(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the DataCollection from outputOfStep2ForStep3 results of the SVD algorithm.
     * @param     id    Identifier of the partial result
     * @param     idx   Index of the DataCollection within KeyValueDataCollcetion of the partial result
     * @return          Value that corresponds to the given identifier and index
     */
    public DataCollection get(DistributedPartialResultCollectionId id, int idx) {
        return (DataCollection)get(id).get(idx);
    }

    /**
     * Returns results of the SVD algorithm with singular values and the left orthogonal matrix calculated
     * @param id    Identifier of the parameter
     * @return      Parameter that corresponds to the given identifier
     */
    public Result get(DistributedPartialResultId id) {
        if (id == DistributedPartialResultId.finalResultFromStep2Master) {
            return new Result(getContext(), cGetResult(getCObject(), id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetKeyValueDataCollection(long presAddr, int id);

    private native long cGetResult(long presAddr, int id);
}
/** @} */
