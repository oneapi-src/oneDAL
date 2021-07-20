/* file: tsne_batch.cpp */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

/*
!  Content:
!    C++ example of tsne gradient descent computarion
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-TSNE-BATCH"></a>
 * \example tsne_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "algorithms/tsne/tsne_gradient_descent.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::internal;

const string initDataset = "../data/batch/tsne_init.csv";
const string pDataset   = "../data/batch/tsne_p.csv";

int main(int argc, char * argv[])
{
	NumericTablePtr initData;
	FileDataSource<CSVFeatureManager> initDataSource(initDataset,
		DataSource::notAllocateNumericTable,
		DataSource::doDictionaryFromContext);
	initData.reset(new HomogenNumericTable<float>(2, 0, NumericTable::notAllocate));
	NumericTablePtr initMerge(new MergedNumericTable(initData));
	initDataSource.loadDataBlock(initMerge.get());

    CSRNumericTablePtr pTable(createSparseTable<float>(pDataset));

	int sizeIter[4] = { 1797, 205798, 300, 1000 }; // nSamples, nnz, nIterWithoutProgress, maxIter
	float params[4] = { 12., 200., 1e-07, 0.5 }; // earlyExaggeration, learningRate, minGradNorm, angle
	float results[3] = { 0., 0., 0. }; // curIter, divergence, gradNorm

	NumericTablePtr sizeIterTable = HomogenNumericTable<int>::create(sizeIter, 1, 4);
	NumericTablePtr paramTable = HomogenNumericTable<float>::create(params, 1, 4);
	NumericTablePtr resultTable = HomogenNumericTable<float>::create(results, 1, 3);

	tsneGradientDescent<int, float>(
		initData,
		pTable,
		sizeIterTable,
		paramTable,
		resultTable);

	cout << "Kullback-Leibler divergence = " << results[1] << endl;

	return 0;
}
