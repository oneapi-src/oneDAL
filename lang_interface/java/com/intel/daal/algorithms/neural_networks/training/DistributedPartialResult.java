/* file: DistributedPartialResult.java */
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
 * @defgroup neural_networks_training_distributed Distributed
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        neural network algorithm in the distributed processing mode on step 2
 */
public class DistributedPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs empty DistributedPartialResult
     * @param context      Context to manage the partial result for the neural network training algorithm
     */
    public DistributedPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public DistributedPartialResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns the result of the neural network training algorithm
     * @param id    Identifier of the partial result
     * @return      TrainingResult that corresponds to the given identifier
     */
    public TrainingResult get(DistributedPartialResultId id) {
        if (id == DistributedPartialResultId.resultFromMaster) {
            return new TrainingResult(getContext(), cGetResult(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewPartialResult();
    private native long cGetResult(long cPartialResult, int id);
}
/** @} */
