/* file: service.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
!    Auxiliary functions used in C++ samples
!******************************************************************************/

#ifndef _SERVICE_H
#define _SERVICE_H

#include "daal.h"

using namespace daal::data_management;

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <vector>
#include <queue>

#include "error_handling.h"

size_t readTextFile(const std::string & datasetFileName, daal::byte ** data)
{
    std::ifstream file(datasetFileName.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        fileOpenError(datasetFileName.c_str());
    }

    std::streampos end = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t fileSize = static_cast<size_t>(end);

    (*data) = new daal::byte[fileSize];
    checkAllocation(data);

    if (!file.read((char *)(*data), fileSize))
    {
        delete[] data;
        fileReadError();
    }

    return fileSize;
}

template <typename item_type>
void readRowUnknownLength(char * line, std::vector<item_type> & data)
{
    size_t n               = 0;
    const char * prevDelim = line - 1;
    char * ptr             = line;
    for (; *ptr; ++ptr)
    {
        if (*ptr == ',' || *ptr == '\r')
        {
            if (prevDelim != ptr - 1) ++n;
            *ptr      = ' ';
            prevDelim = ptr;
        }
    }
    if (prevDelim != ptr - 1) ++n;
    data.resize(n);
    std::stringstream iss(line);
    for (size_t i = 0; i < n; ++i)
    {
        iss >> data[i];
    }
}

template <typename item_type>
CSRNumericTable * createSparseTable(const std::string & datasetFileName)
{
    std::ifstream file(datasetFileName.c_str());

    if (!file.is_open())
    {
        fileOpenError(datasetFileName.c_str());
    }

    std::string str;

    //read row offsets
    std::getline(file, str);
    std::vector<size_t> rowOffsets;
    readRowUnknownLength<size_t>(&str[0], rowOffsets);
    if (!rowOffsets.size()) return NULL;
    const size_t nVectors = rowOffsets.size() - 1;

    //read cols indices
    std::getline(file, str);
    std::vector<size_t> colIndices;
    readRowUnknownLength<size_t>(&str[0], colIndices);

    //read values
    std::getline(file, str);
    std::vector<item_type> data;
    readRowUnknownLength<item_type>(&str[0], data);
    const size_t nNonZeros = data.size();

    size_t maxCol = 0;
    for (size_t i = 0; i < colIndices.size(); ++i)
    {
        if (colIndices[i] > maxCol) maxCol = colIndices[i];
    }
    const size_t nFeatures = maxCol;

    if (!nFeatures || !nVectors || colIndices.size() != nNonZeros || nNonZeros != (rowOffsets[nVectors] - 1))
    {
        sparceFileReadError();
    }

    size_t * resultRowOffsets      = NULL;
    size_t * resultColIndices      = NULL;
    item_type * resultData         = NULL;
    CSRNumericTable * numericTable = new CSRNumericTable(resultData, resultColIndices, resultRowOffsets, nFeatures, nVectors);
    numericTable->allocateDataMemory(nNonZeros);
    numericTable->getArrays<item_type>(&resultData, &resultColIndices, &resultRowOffsets);
    for (size_t i = 0; i < nNonZeros; ++i)
    {
        resultData[i]       = data[i];
        resultColIndices[i] = colIndices[i];
    }
    for (size_t i = 0; i < nVectors + 1; ++i)
    {
        resultRowOffsets[i] = rowOffsets[i];
    }
    return numericTable;
}

void printAprioriItemsets(NumericTablePtr largeItemsetsTable, NumericTablePtr largeItemsetsSupportTable, size_t nItemsetToPrint = 20)
{
    size_t largeItemsetCount     = largeItemsetsSupportTable->getNumberOfRows();
    size_t nItemsInLargeItemsets = largeItemsetsTable->getNumberOfRows();

    BlockDescriptor<int> block1;
    largeItemsetsTable->getBlockOfRows(0, nItemsInLargeItemsets, readOnly, block1);
    int * largeItemsets = block1.getBlockPtr();

    BlockDescriptor<int> block2;
    largeItemsetsSupportTable->getBlockOfRows(0, largeItemsetCount, readOnly, block2);
    int * largeItemsetsSupportData = block2.getBlockPtr();

    std::vector<std::vector<size_t> > largeItemsetsVector;
    largeItemsetsVector.resize(largeItemsetCount);

    for (size_t i = 0; i < nItemsInLargeItemsets; i++)
    {
        largeItemsetsVector[largeItemsets[2 * i]].push_back(largeItemsets[2 * i + 1]);
    }

    std::vector<size_t> supportVector;
    supportVector.resize(largeItemsetCount);

    for (size_t i = 0; i < largeItemsetCount; i++)
    {
        supportVector[largeItemsetsSupportData[2 * i]] = largeItemsetsSupportData[2 * i + 1];
    }

    std::cout << std::endl << "Apriori example program results" << std::endl;

    std::cout << std::endl << "Last " << nItemsetToPrint << " large itemsets: " << std::endl;
    std::cout << std::endl
              << "Itemset"
              << "\t\t\tSupport" << std::endl;

    size_t iMin = (((largeItemsetCount > nItemsetToPrint) && (nItemsetToPrint != 0)) ? largeItemsetCount - nItemsetToPrint : 0);
    for (size_t i = iMin; i < largeItemsetCount; i++)
    {
        std::cout << "{";
        for (size_t l = 0; l < largeItemsetsVector[i].size() - 1; l++)
        {
            std::cout << largeItemsetsVector[i][l] << ", ";
        }
        std::cout << largeItemsetsVector[i][largeItemsetsVector[i].size() - 1] << "}\t\t";

        std::cout << supportVector[i] << std::endl;
    }

    largeItemsetsTable->releaseBlockOfRows(block1);
    largeItemsetsSupportTable->releaseBlockOfRows(block2);
}

void printAprioriRules(NumericTablePtr leftItemsTable, NumericTablePtr rightItemsTable, NumericTablePtr confidenceTable, size_t nRulesToPrint = 20)
{
    size_t nRules      = confidenceTable->getNumberOfRows();
    size_t nLeftItems  = leftItemsTable->getNumberOfRows();
    size_t nRightItems = rightItemsTable->getNumberOfRows();

    BlockDescriptor<int> block1;
    leftItemsTable->getBlockOfRows(0, nLeftItems, readOnly, block1);
    int * leftItems = block1.getBlockPtr();

    BlockDescriptor<int> block2;
    rightItemsTable->getBlockOfRows(0, nRightItems, readOnly, block2);
    int * rightItems = block2.getBlockPtr();

    BlockDescriptor<DAAL_DATA_TYPE> block3;
    confidenceTable->getBlockOfRows(0, nRules, readOnly, block3);
    DAAL_DATA_TYPE * confidence = block3.getBlockPtr();

    std::vector<std::vector<size_t> > leftItemsVector;
    leftItemsVector.resize(nRules);

    if (nRules == 0)
    {
        std::cout << std::endl << "No association rules were found " << std::endl;
        return;
    }

    for (size_t i = 0; i < nLeftItems; i++)
    {
        leftItemsVector[leftItems[2 * i]].push_back(leftItems[2 * i + 1]);
    }

    std::vector<std::vector<size_t> > rightItemsVector;
    rightItemsVector.resize(nRules);

    for (size_t i = 0; i < nRightItems; i++)
    {
        rightItemsVector[rightItems[2 * i]].push_back(rightItems[2 * i + 1]);
    }

    std::vector<DAAL_DATA_TYPE> confidenceVector;
    confidenceVector.resize(nRules);

    for (size_t i = 0; i < nRules; i++)
    {
        confidenceVector[i] = confidence[i];
    }

    std::cout << std::endl << "Last " << nRulesToPrint << " association rules: " << std::endl;
    std::cout << std::endl
              << "Rule"
              << "\t\t\t\tConfidence" << std::endl;
    size_t iMin = (((nRules > nRulesToPrint) && (nRulesToPrint != 0)) ? (nRules - nRulesToPrint) : 0);

    for (size_t i = iMin; i < nRules; i++)
    {
        std::cout << "{";
        for (size_t l = 0; l < leftItemsVector[i].size() - 1; l++)
        {
            std::cout << leftItemsVector[i][l] << ", ";
        }
        std::cout << leftItemsVector[i][leftItemsVector[i].size() - 1] << "} => {";

        for (size_t l = 0; l < rightItemsVector[i].size() - 1; l++)
        {
            std::cout << rightItemsVector[i][l] << ", ";
        }
        std::cout << rightItemsVector[i][rightItemsVector[i].size() - 1] << "}\t\t";

        std::cout << confidenceVector[i] << std::endl;
    }

    leftItemsTable->releaseBlockOfRows(block1);
    rightItemsTable->releaseBlockOfRows(block2);
    confidenceTable->releaseBlockOfRows(block3);
}

bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int)layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix || layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix || layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename T>
void printArray(T * array, const size_t nPrintedCols, const size_t nPrintedRows, const size_t nCols, std::string message, size_t interval = 10)
{
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    for (size_t i = 0; i < nPrintedRows; i++)
    {
        for (size_t j = 0; j < nPrintedCols; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << array[i * nCols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void printArray(T * array, const size_t nCols, const size_t nRows, std::string message, size_t interval = 10)
{
    printArray(array, nCols, nRows, nCols, message, interval);
}

template <typename T>
void printLowerArray(T * array, const size_t nPrintedRows, std::string message, size_t interval = 10)
{
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++)
    {
        for (size_t j = 0; j <= i; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << array[ind++];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void printUpperArray(T * array, const size_t nPrintedCols, const size_t nPrintedRows, const size_t nCols, std::string message, size_t interval = 10)
{
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            std::cout << "          ";
        }
        for (size_t j = i; j < nPrintedCols; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << array[ind++];
        }
        for (size_t j = nPrintedCols; j < nCols; j++)
        {
            ind++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printNumericTable(NumericTable * dataTable, const char * message = "", size_t nPrintedRows = 0, size_t nPrintedCols = 0, size_t interval = 10)
{
    size_t nRows                            = dataTable->getNumberOfRows();
    size_t nCols                            = dataTable->getNumberOfColumns();
    NumericTableIface::StorageLayout layout = dataTable->getDataLayout();

    if (nPrintedRows != 0)
    {
        nPrintedRows = std::min(nRows, nPrintedRows);
    }
    else
    {
        nPrintedRows = nRows;
    }

    if (nPrintedCols != 0)
    {
        nPrintedCols = std::min(nCols, nPrintedCols);
    }
    else
    {
        nPrintedCols = nCols;
    }

    BlockDescriptor<DAAL_DATA_TYPE> block;
    if (isFull(layout) || layout == NumericTableIface::csrArray)
    {
        dataTable->getBlockOfRows(0, nRows, readOnly, block);
        printArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);
        dataTable->releaseBlockOfRows(block);
    }
    else
    {
        PackedArrayNumericTableIface * packedTable = dynamic_cast<PackedArrayNumericTableIface *>(dataTable);
        packedTable->getPackedArray(readOnly, block);
        if (isLower(layout))
        {
            printLowerArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedRows, message, interval);
        }
        else if (isUpper(layout))
        {
            printUpperArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);
        }
        packedTable->releasePackedArray(block);
    }
}

void printNumericTable(NumericTable & dataTable, const char * message = "", size_t nPrintedRows = 0, size_t nPrintedCols = 0, size_t interval = 10)
{
    printNumericTable(&dataTable, message, nPrintedRows, nPrintedCols, interval);
}

void printNumericTable(const NumericTablePtr & dataTable, const char * message = "", size_t nPrintedRows = 0, size_t nPrintedCols = 0,
                       size_t interval = 10)
{
    printNumericTable(dataTable.get(), message, nPrintedRows, nPrintedCols, interval);
}

void printPackedNumericTable(NumericTable * dataTable, size_t nFeatures, const char * message = "", size_t interval = 10)
{
    BlockDescriptor<DAAL_DATA_TYPE> block;

    dataTable->getBlockOfRows(0, 1, readOnly, block);

    DAAL_DATA_TYPE * data = block.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    size_t index = 0;
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j <= i; j++, index++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << data[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable->releaseBlockOfRows(block);
}

void printPackedNumericTable(NumericTable & dataTable, size_t nFeatures, const char * message = "", size_t interval = 10)
{
    printPackedNumericTable(&dataTable, nFeatures, message);
}

template <typename type1, typename type2>
void printNumericTables(NumericTable * dataTable1, NumericTable * dataTable2, const char * title1 = "", const char * title2 = "",
                        const char * message = "", size_t nPrintedRows = 0, size_t interval = 15)
{
    size_t nRows1 = dataTable1->getNumberOfRows();
    size_t nRows2 = dataTable2->getNumberOfRows();
    size_t nCols1 = dataTable1->getNumberOfColumns();
    size_t nCols2 = dataTable2->getNumberOfColumns();

    BlockDescriptor<type1> block1;
    BlockDescriptor<type2> block2;

    size_t nRows = std::min(nRows1, nRows2);
    if (nPrintedRows != 0)
    {
        nRows = std::min(std::min(nRows1, nRows2), nPrintedRows);
    }

    dataTable1->getBlockOfRows(0, nRows, readOnly, block1);
    dataTable2->getBlockOfRows(0, nRows, readOnly, block2);

    type1 * data1 = block1.getBlockPtr();
    type2 * data2 = block2.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    std::cout << std::setw(interval * nCols1) << title1;
    std::cout << std::setw(interval * nCols2) << title2 << std::endl;
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols1; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << data1[i * nCols1 + j];
        }
        for (size_t j = 0; j < nCols2; j++)
        {
            std::cout << std::setprecision(0) << std::setw(interval) << data2[i * nCols2 + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable1->releaseBlockOfRows(block1);
    dataTable2->releaseBlockOfRows(block2);
}

template <typename type1, typename type2>
void printNumericTables(NumericTable * dataTable1, NumericTable & dataTable2, const char * title1 = "", const char * title2 = "",
                        const char * message = "", size_t nPrintedRows = 0, size_t interval = 10)
{
    printNumericTables<type1, type2>(dataTable1, &dataTable2, title1, title2, message, nPrintedRows, interval);
}

void printNumericTables(NumericTable * dataTable1, NumericTable * dataTable2, const char * title1 = "", const char * title2 = "",
                        const char * message = "", size_t nPrintedRows = 0, size_t interval = 10)
{
    size_t nRows1 = dataTable1->getNumberOfRows();
    size_t nRows2 = dataTable2->getNumberOfRows();
    size_t nCols1 = dataTable1->getNumberOfColumns();
    size_t nCols2 = dataTable2->getNumberOfColumns();

    BlockDescriptor<DAAL_DATA_TYPE> block1;
    BlockDescriptor<DAAL_DATA_TYPE> block2;

    size_t nRows = std::min(nRows1, nRows2);
    if (nPrintedRows != 0)
    {
        nRows = std::min(std::min(nRows1, nRows2), nPrintedRows);
    }

    dataTable1->getBlockOfRows(0, nRows, readOnly, block1);
    dataTable2->getBlockOfRows(0, nRows, readOnly, block2);

    DAAL_DATA_TYPE * data1 = block1.getBlockPtr();
    DAAL_DATA_TYPE * data2 = block2.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    std::cout << std::setw(interval * nCols1) << title1;
    std::cout << std::setw(interval * nCols2) << title2 << std::endl;
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols1; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << data1[i * nCols1 + j];
        }
        for (size_t j = 0; j < nCols2; j++)
        {
            std::cout << std::setprecision(0) << std::setw(interval) << data2[i * nCols2 + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable1->releaseBlockOfRows(block1);
    dataTable2->releaseBlockOfRows(block2);
}

void printNumericTables(NumericTable * dataTable1, NumericTable & dataTable2, const char * title1 = "", const char * title2 = "",
                        const char * message = "", size_t nPrintedRows = 0, size_t interval = 10)
{
    printNumericTables(dataTable1, &dataTable2, title1, title2, message, nPrintedRows, interval);
}

template <typename type1, typename type2>
void printNumericTables(NumericTablePtr dataTable1, NumericTablePtr dataTable2, const char * title1 = "", const char * title2 = "",
                        const char * message = "", size_t nPrintedRows = 0, size_t interval = 10)
{
    printNumericTables<type1, type2>(dataTable1.get(), dataTable2.get(), title1, title2, message, nPrintedRows, interval);
}

bool checkFileIsAvailable(std::string filename, bool needExit = false)
{
    std::ifstream file(filename.c_str());
    if (file.good())
    {
        return true;
    }
    else
    {
        std::cout << "Can't open file " << filename << std::endl;
        if (needExit)
        {
            exit(fileError);
        }
        return false;
    }
}

void checkArguments(int argc, char * argv[], int count, ...)
{
    std::string ** filelist = new std::string *[count];
    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++)
    {
        filelist[i] = va_arg(ap, std::string *);
    }
    va_end(ap);
    if (argc == 1)
    {
        for (int i = 0; i < count; i++)
        {
            checkFileIsAvailable(*(filelist[i]), true);
        }
    }
    else if (argc == (count + 1))
    {
        bool isAllCorrect = true;
        for (int i = 0; i < count; i++)
        {
            if (!checkFileIsAvailable(argv[i + 1]))
            {
                isAllCorrect = false;
                break;
            }
        }
        if (isAllCorrect == true)
        {
            for (int i = 0; i < count; i++)
            {
                (*filelist[i]) = argv[i + 1];
            }
        }
        else
        {
            std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
            for (int i = 0; i < count; i++)
            {
                checkFileIsAvailable(*(filelist[i]), true);
            }
        }
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [ ";
        for (int i = 0; i < count; i++)
        {
            std::cout << "<filename_" << i << "> ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
        for (int i = 0; i < count; i++)
        {
            checkFileIsAvailable(*(filelist[i]), true);
        }
    }
    delete[] filelist;
}

void copyBytes(daal::byte * dst, daal::byte * src, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        dst[i] = src[i];
    }
}

size_t checkBytes(daal::byte * dst, daal::byte * src, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (dst[i] != src[i])
        {
            return i + 1;
        }
    }
    return 0;
}

#endif
