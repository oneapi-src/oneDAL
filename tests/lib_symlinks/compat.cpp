#include "daal.h"

using namespace daal::algorithms;
using namespace daal::data_management;

int main(int argc, char const *argv[]) {
    const std::size_t row_count = 8;
    const std::size_t column_count = 7;
    const std::size_t cluster_count = 2;
    const std::size_t max_iteration_count = 5;

    float data_raw[] = {
        1.f,  2.f,  3.f,  4.f,  5.f,  6.f, -5.f,
        1.f, -1.f,  0.f,  3.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  1.f,  0.f,  0.f,  0.f,
        1.f,  2.f,  5.f,  2.f,  9.f,  3.f,  2.f,
       -4.f,  3.f,  0.f,  4.f,  2.f,  7.f,  5.f,
        4.f,  2.f,  0.f, -4.f,  0.f,  3.f, -8.f,
        2.f,  5.f,  5.f, -6.f,  3.f,  0.f, -9.f,
        3.f,  1.f, -3.f,  3.f,  5.f,  1.f,  7.f
    };

    const auto data = HomogenNumericTable<float>::create(data_raw, column_count, row_count);

    kmeans::init::Batch<float, kmeans::init::randomDense> init(cluster_count);
    init.input.set(kmeans::init::data, data);
    init.compute();

    const auto centroids = init.getResult()->get(kmeans::init::centroids);

    kmeans::Batch<> algorithm(cluster_count, max_iteration_count);
    algorithm.input.set(kmeans::data, data);
    algorithm.input.set(kmeans::inputCentroids, centroids);
    algorithm.parameter().resultsToEvaluate = kmeans::computeCentroids |
                                              kmeans::computeAssignments |
                                              kmeans::computeExactObjectiveFunction;
    algorithm.compute();

    return 0;
}
