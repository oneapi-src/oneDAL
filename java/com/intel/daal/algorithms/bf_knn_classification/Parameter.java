/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup bf_knn_classification
 * @{
 */
package com.intel.daal.algorithms.bf_knn_classification;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PARAMETER"></a>
 * @brief brute-force k nearest neighbors algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the number of neighbors
     * @param k  Number of neighbors
     */
    public void setK(long k) {
        cSetK(this.cObject, k);
    }

    /**
     * Returns the number of neighbors
     * @return Number of neighbors
     */
    public long getK() {
        return cGetK(this.cObject);
    }

    /**
     * @DAAL_DEPRECATED
     * Sets the weight function used in prediction voting
     * @param voteWeights   Weight function used in prediction voting
     */
    public void setVoteWeights(VoteWeightsId voteWeights) {
        cSetVoteWeights(this.cObject, voteWeights.getValue());
    }

    /**
     * @DAAL_DEPRECATED
     * Returns the weight function used in prediction voting
     * @return Weight function used in prediction voting
     */
    public VoteWeightsId getVoteWeights() {
        return new VoteWeightsId(cGetVoteWeights(this.cObject));
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    /**
     * Sets the enable/disable an usage of the input dataset in kNN model flag
     * @param flag   Enable/disable an usage of the input dataset in kNN model flag
     */
    public void setDataUseInModel(DataUseInModelId flag) {
        cSetDataUseInModel(this.cObject, flag.getValue());
    }

    /**
     * Returns the enable/disable an usage of the input dataset in kNN model flag
     * @return Enable/disable an usage of the input dataset in kNN model flag
     */
    public DataUseInModelId getDataUseInModel() {
        return new DataUseInModelId(cGetDataUseInModel(this.cObject));
    }

    /**
     * Sets the flag that indicates the results to compute
     * @param resultsToCompute   Flag that indicates the results to compute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute);
    }

    /**
     * Returns the flag that indicates the results to compute
     * @return Flag that indicates the results to compute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject);
    }

    private native void cSetK(long algAddr, long k);
    private native void cSetVoteWeights(long algAddr, int flag);
    private native void cSetEngine(long cObject, long cEngineObject);
    private native void cSetDataUseInModel(long algAddr, int flag);
    private native void cSetResultsToCompute(long algAddr, long flag);

    private native long cGetK(long algAddr);
    private native int cGetVoteWeights(long algAddr);
    private native int cGetDataUseInModel(long algAddr);
    private native long cGetResultsToCompute(long algAddr);
}
/** @} */
