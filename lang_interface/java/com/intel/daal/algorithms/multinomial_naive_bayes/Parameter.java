/* file: Parameter.java */
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
 * @brief Contains classes for computing the Naive Bayes
 */
package com.intel.daal.algorithms.multinomial_naive_bayes;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARAMETER"></a>
 * @brief Parameters for multinomial naive Bayes algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets prior class estimates, numeric table of size [nClasses x 1]
     *  @param priorClassEstimates  Prior class estimates
     */
    public void setPriorClassEstimates(NumericTable priorClassEstimates) {
        cSetPriorClassEstimates(this.cObject, priorClassEstimates.getCObject());
    }

    /**
     *  Returnss prior class estimates, numeric table of size [nClasses x 1]
     *  @return  Prior class estimates
     */
    public NumericTable getPriorClassEstimates() {
        NumericTable nt = new HomogenNumericTable(getContext(), cGetPriorClassEstimates(this.cObject));
        return nt;
    }

    /**
     *  Sets imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     *  @param alpha  Imagined occurrences of the each feature
     */
    public void setAlpha(NumericTable alpha) {
        cSetAlpha(this.cObject, alpha.getCObject());
    }

    /**
     *  Returnss imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     *  @return  Imagined occurrences of the each feature
     */
    public NumericTable getAlpha() {
        NumericTable nt = new HomogenNumericTable(getContext(), cGetAlpha(this.cObject));
        return nt;
    }

    private native void cSetPriorClassEstimates(long parAddr, long ntAddr);

    private native long cGetPriorClassEstimates(long parAddr);

    private native void cSetAlpha(long parAddr, long ntAddr);

    private native long cGetAlpha(long parAddr);
}
