/* file: DistributedPartialResult.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @defgroup neural_networks_training_distributed Distributed
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

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
        System.loadLibrary("JavaAPI");
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
