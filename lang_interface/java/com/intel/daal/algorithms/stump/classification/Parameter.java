/* file: Parameter.java */
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
 * @ingroup stump_classification
 */
/**
 * @brief Contains classes of the stump classification algorithm
 */
package com.intel.daal.algorithms.stump.classification;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.classification.SplitCriterionId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__CLASSIFICATION__PARAMETER"></a>
 * @brief Base class for parameters of the decision stump classification algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {
    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Retrieves the split criterion for stump classification
     * @return Split criterion for stump classification
     */
    public SplitCriterionId getSplitCriterion() {
        return new SplitCriterionId(cGetSplitCriterion(this.cObject));
    }

    /**
     * Sets the split criterion for stump classification
     * @param splitCriterion Split criterion for stump classification
     */
    public void setSplitCriterion(SplitCriterionId splitCriterion) {
        cSetSplitCriterion(this.cObject, splitCriterion.getValue());
    }

    /**
     * Retrieves the variable importance computation mode
     * @return Variable importance computation mode
     */
    public VariableImportanceModeId getVarImportance() {
        return new VariableImportanceModeId(cGetVarImportance(this.cObject));
    }

    /**
     * Sets the variable importance computation mode
     * @param variableImportanceMode Variable importance computation mode
     */
    public void setVarImportance(VariableImportanceModeId variableImportanceMode) {
        VariableImportanceModeId.throwIfInvalid(variableImportanceMode);
        cSetVarImportance(this.cObject, variableImportanceMode.getValue());
    }

    private native int cGetSplitCriterion(long selfPtr);
    private native void cSetSplitCriterion(long selfPtr, int splitCriterion);

    private native int cGetVarImportance(long selfPtr);
    private native void cSetVarImportance(long selfPtr, int variableImportanceMode);
}
/** @} */
