/* file: service.h */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    Auxiliary functions used in C++ neural networks samples
!******************************************************************************/

#ifndef _SERVICE_H
#define _SERVICE_H

#include "daal_defines.h"

#include <string>
#include <cstdarg>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

void printPredictedClasses(const TensorPtr &prediction, const TensorPtr &testingGroundTruth, size_t maxRows=0)
{
    const Collection<size_t> &predictionDimensions = prediction->getDimensions();

    SubtensorDescriptor<DAAL_DATA_TYPE> predictionBlock;
    prediction->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, predictionBlock);
    DAAL_DATA_TYPE *predictionPtr = predictionBlock.getPtr();

    SubtensorDescriptor<int> testGroundTruthBlock;
    testingGroundTruth->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, testGroundTruthBlock);
    int *testGroundTruthPtr = testGroundTruthBlock.getPtr();

    size_t requiredRows = maxRows > 0 ? maxRows : predictionDimensions[0];
    size_t rowsNum = std::min(requiredRows, predictionDimensions[0]);
    for (size_t i = 0; i < rowsNum; i++)
    {
        DAAL_DATA_TYPE maxP = 0;
        size_t maxPIndex = 0;
        for (size_t j = 0; j < predictionDimensions[1]; j++)
        {
            DAAL_DATA_TYPE p = predictionPtr[i * predictionDimensions[1] + j];
            if (maxP < p)
            {
                maxP = p;
                maxPIndex = j;
            }
            printf("%.4f ", p);
        }
        fflush(stdout);

        printf(" -> %d | %d\n", maxPIndex, testGroundTruthPtr[i]);
        fflush(stdout);
    }

    prediction->releaseSubtensor(predictionBlock);
    testingGroundTruth->releaseSubtensor(testGroundTruthBlock);
}

struct ArrayIndicesComparerByProbability
{
    ArrayIndicesComparerByProbability(float *probabilities) :
        _probabilities(probabilities) { }

    bool operator ()(const size_t &i, const size_t &j)
    {
        return _probabilities[i] < _probabilities[j];
    }

private:
    float *_probabilities;
};

class ClassificationErrorCounter
{
private:
    static const size_t MAX_ERROR_RATE_CLASSES;

    size_t _totalObjects;
    size_t _top1ClassifiedObjects;
    size_t _top5ClassifiedObjects;

public:
    ClassificationErrorCounter() { initialize(); }

    ClassificationErrorCounter(const TensorPtr &prediction, const TensorPtr &groundTruth)
    {
        initialize();
        update(prediction, groundTruth);
    }

    void update(const TensorPtr &prediction,
                const TensorPtr &groundTruth);

    inline float getTop1ErrorRate() const { return getErrorRate(_top1ClassifiedObjects); }

    inline float getTop5ErrorRate() const { return getErrorRate(_top5ClassifiedObjects); }

private:
    inline void initialize()
    {
        _totalObjects = 0;
        _top1ClassifiedObjects = 0;
        _top5ClassifiedObjects = 0;
    }

    inline float getErrorRate(size_t topClassifiedObjectsNum) const
    {
        if (_totalObjects == 0) { return 1.0f; }
        return 1.f - topClassifiedObjectsNum / (float)_totalObjects;
    }

    static void findTopIndices(float *rowPtr, size_t colsNum, size_t *topIndices, size_t topSize);

    static bool isValueInArray(size_t *array, size_t size, size_t value);
};

const size_t ClassificationErrorCounter::MAX_ERROR_RATE_CLASSES = 5;

void ClassificationErrorCounter::update(const TensorPtr &prediction,
                                        const TensorPtr &groundTruth)
{
    if (!prediction)  { throw std::runtime_error("Prediction tensor is not able to be null");  }
    if (!groundTruth) { throw std::runtime_error("GroundTruth tensor is not able to be null"); }

    Collection<size_t> dimensions = prediction->getDimensions();
    if (dimensions.size() != 2) { throw std::runtime_error("Predictions tensor should have exactly two dimensions"); }
    const size_t rowsNum = dimensions[0];
    const size_t colsNum = dimensions[1];

    if (colsNum < MAX_ERROR_RATE_CLASSES)
    {
        throw std::runtime_error("Number of classes in prediction result "
                                 "is not enough to compute error rate");
    }

    SubtensorDescriptor<DAAL_DATA_TYPE> predictionBlock;
    prediction->getSubtensor(0, 0, 0, dimensions[0], readOnly, predictionBlock);
    DAAL_DATA_TYPE *predictionPtr = predictionBlock.getPtr();

    SubtensorDescriptor<int> groundTruthBlock;
    groundTruth->getSubtensor(0, 0, 0, dimensions[0], readOnly, groundTruthBlock);
    int *groundTruthPtr = groundTruthBlock.getPtr();

    size_t *topIndices = new size_t[MAX_ERROR_RATE_CLASSES];
    for (size_t i = 0; i < rowsNum; i++)
    {
        DAAL_DATA_TYPE *rowPtr = predictionPtr + i * colsNum;
        findTopIndices(rowPtr, colsNum, topIndices, MAX_ERROR_RATE_CLASSES);

        size_t groundTruthClass = (size_t)groundTruthPtr[i];
        bool isInTopFound = isValueInArray(topIndices, MAX_ERROR_RATE_CLASSES, groundTruthClass);

        _totalObjects++;
        if (isInTopFound)
        {
            _top5ClassifiedObjects++;
            if (groundTruthClass == topIndices[0])
            {
                _top1ClassifiedObjects++;
            }
        }
    }
    delete[] topIndices;

    prediction->releaseSubtensor(predictionBlock);
    groundTruth->releaseSubtensor(groundTruthBlock);
}

void ClassificationErrorCounter::findTopIndices(float *rowPtr, size_t colsNum,
                                                size_t *topIndices, size_t topSize)
{
    size_t *indices = new size_t[colsNum];
    for (size_t i = 0; i < colsNum; i++)
    {
        indices[i] = i;
    }

    std::sort(indices, indices + colsNum, ArrayIndicesComparerByProbability(rowPtr));

    for (size_t i = 0; i < topSize; ++i)
    {
        topIndices[i] = indices[colsNum - i - 1];
    }

    delete[] indices;
}

bool ClassificationErrorCounter::isValueInArray(size_t *array, size_t size, size_t value)
{
    size_t *begin = array;
    size_t *end = array + size;
    return std::find(begin, end, value) != end;
}


bool checkFileIsAvailable(const std::string &filename)
{
    std::ifstream file(filename.c_str());
    return file.good();
}

bool checkFilesAreAvailable(const std::string &basePath,
                            const std::string *datasetFileNames,
                            size_t numberOfFiles)
{
    std::string filePath;
    for (size_t i = 0; i < numberOfFiles; i++)
    {
        filePath.clear();
        if (basePath.size())
        {
            filePath = basePath + "/";
        }
        filePath += datasetFileNames[i];
        if (!checkFileIsAvailable(filePath))
        {
            return false;
        }
    }
    return true;
}

std::string selectDatasetPathOrExit(const std::string &defaultDatasetsPath,
                                    const std::string &userDatasetsPath,
                                    const std::string *datasetFileNames,
                                    size_t numberOfFiles)
{
    if (userDatasetsPath.size() && checkFilesAreAvailable(userDatasetsPath, datasetFileNames, numberOfFiles))
    {
        return userDatasetsPath;
    }
    else if (checkFilesAreAvailable(defaultDatasetsPath, datasetFileNames, numberOfFiles))
    {
        if (userDatasetsPath.size())
        {
            std::cout << "Warning: Can't open dataset from path provided via command line: ";
            std::cout << userDatasetsPath << std::endl;
            std::cout << "         Try to open dataset from default path: ";
            std::cout << defaultDatasetsPath << std::endl;
        }
        return defaultDatasetsPath;
    }

    std::cout << "Error: Can't open datasets from default path: " << defaultDatasetsPath << std::endl;
    exit(-1);
}


std::string getUserDatasetPath(int argc, char *argv[])
{
    std::string userDatasetsPath;
    if (argc > 1) { userDatasetsPath = argv[1]; }
    return userDatasetsPath;
}

TensorPtr readTensorFromCSV(const std::string &datasetFileName, bool allowOneColumn = false)
{
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();

    daal::services::SharedPtr<HomogenNumericTable<DAAL_DATA_TYPE> > ntPtr =
        daal::services::staticPointerCast<HomogenNumericTable<DAAL_DATA_TYPE>, NumericTable>(dataSource.getNumericTable());

    daal::services::Collection<size_t> dims;
    dims.push_back(ntPtr->getNumberOfRows());
    size_t size = dims[0];
    if (ntPtr->getNumberOfColumns() > 1 || allowOneColumn)
    {
        dims.push_back(ntPtr->getNumberOfColumns());
        size *= dims[1];
    }

    HomogenTensor<DAAL_DATA_TYPE> *tensor = new HomogenTensor<DAAL_DATA_TYPE>( dims, Tensor::doAllocate );
    DAAL_DATA_TYPE *tensorData = tensor->getArray();
    DAAL_DATA_TYPE *ntData = ntPtr->getArray();

    for(size_t i = 0; i < size; i++)
    {
        tensorData[i] = ntData[i];
    }

    TensorPtr tensorPtr(tensor);

    return tensorPtr;
}

template <typename T>
void printArray(T *array, const size_t nPrintedCols, const size_t nPrintedRows, const size_t nCols, std::string message,
                size_t interval = 10)
{
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    for (size_t i = 0; i < nPrintedRows; i++)
    {
        for(size_t j = 0; j < nPrintedCols; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << array[i * nCols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printTensor(TensorPtr dataTable, const char *message = "",
                 size_t nPrintedRows = 0, size_t nPrintedCols = 0, size_t interval = 10)
{
    const daal::services::Collection<size_t> &dims = dataTable->getDimensions();
    size_t nRows = dims[0];

    if(nPrintedRows != 0)
    {
        nPrintedRows = std::min(nRows, nPrintedRows);
    }
    else
    {
        nPrintedRows = nRows;
    }

    SubtensorDescriptor<DAAL_DATA_TYPE> block;

    dataTable->getSubtensor(0, 0, 0, nPrintedRows, readOnly, block);

    size_t nCols = block.getSize() / nPrintedRows;

    if(nPrintedCols != 0)
    {
        nPrintedCols = std::min(nCols, nPrintedCols);
    }
    else
    {
        nPrintedCols = nCols;
    }

    printArray<DAAL_DATA_TYPE>(block.getPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);

    dataTable->releaseSubtensor(block);
}

#endif
