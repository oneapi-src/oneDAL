#include "src/algorithms/dtrees/gbt/treeshap.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace treeshap
{
namespace internal
{

// extend our decision path with a fraction of one and zero extensions
void treeShapExtendPath(PathElement * uniquePath, size_t uniqueDepth, float zeroFraction, float oneFraction, FeatureIndexType featureIndex)
{
    uniquePath[uniqueDepth].featureIndex  = featureIndex;
    uniquePath[uniqueDepth].zeroFraction  = zeroFraction;
    uniquePath[uniqueDepth].oneFraction   = oneFraction;
    uniquePath[uniqueDepth].partialWeight = (uniqueDepth == 0 ? 1.0f : 0.0f);

    for (int i = uniqueDepth - 1; i >= 0; i--)
    {
        uniquePath[i + 1].partialWeight += oneFraction * uniquePath[i].partialWeight * (i + 1) / static_cast<float>(uniqueDepth + 1);
        uniquePath[i].partialWeight = zeroFraction * uniquePath[i].partialWeight * (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
    }
}

// undo a previous extension of the decision path
void treeShapUnwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    printf("treeShapUnwindPath: Going through path elements\n");
    printf("uniquePath  = %p\n", uniquePath);
    printf("uniqueDepth = %lu\n", uniqueDepth);
    printf("pathIndex   = %lu\n", pathIndex);
    printf("---- start\n");
    printf("%p\n", uniquePath + pathIndex);

    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;

    printf("%p\n", uniquePath + uniqueDepth);
    float nextOnePortion = uniquePath[uniqueDepth].partialWeight;

    for (int i = uniqueDepth - 1; i >= 0; --i)
    {
        printf("%p\n", uniquePath + i);
        if (oneFraction != 0)
        {
            const float tmp             = uniquePath[i].partialWeight;
            uniquePath[i].partialWeight = nextOnePortion * (uniqueDepth + 1) / static_cast<float>((i + 1) * oneFraction);
            nextOnePortion              = tmp - uniquePath[i].partialWeight * zeroFraction * (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
        }
        else
        {
            uniquePath[i].partialWeight = (uniquePath[i].partialWeight * (uniqueDepth + 1)) / static_cast<float>(zeroFraction * (uniqueDepth - i));
        }
    }

    for (size_t i = pathIndex; i < uniqueDepth; ++i)
    {
        printf("%p <- %p\n", uniquePath + i, uniquePath + i + 1);
        uniquePath[i].featureIndex = uniquePath[i + 1].featureIndex;
        uniquePath[i].zeroFraction = uniquePath[i + 1].zeroFraction;
        uniquePath[i].oneFraction  = uniquePath[i + 1].oneFraction;
    }
}

// determine what the total permutation weight would be if we unwound a previous extension in the decision path
float treeShapUnwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    printf("treeShapUnwoundPathSum: Going through path elements\n");
    printf("uniquePath  = %p\n", uniquePath);
    printf("uniqueDepth = %lu\n", uniqueDepth);
    printf("pathIndex   = %lu\n", pathIndex);
    printf("---- start\n");
    printf("%p\n", uniquePath + pathIndex);

    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;

    printf("%p\n", uniquePath + uniqueDepth);
    float nextOnePortion = uniquePath[uniqueDepth].partialWeight;
    float total          = 0;
    // if (oneFraction != 0)
    // {
    //     float nextOnePortion = uniquePath[uniqueDepth].partialWeight;
    //     for (int i = uniqueDepth - 1; i >= 0; --i)
    //     {
    //         const float tmp = nextOnePortion * (uniqueDepth + 1) / static_cast<float>((i + 1) * oneFraction);
    //         total += tmp;
    //         nextOnePortion = uniquePath[i].partialWeight - tmp * zeroFraction * ((uniqueDepth - i) / static_cast<float>(uniqueDepth + 1));
    //     }
    // }
    // else if (zeroFraction != 0)
    // {
    //     for (int i = uniqueDepth - 1; i >= 0; --i)
    //     {
    //         total += uniquePath[i].partialWeight * (uniqueDepth + 1) / ((uniqueDepth - i) * zeroFraction);
    //     }
    // }
    // else
    // {
    //     for (int i = uniqueDepth - 1; i >= 0; --i)
    //     {
    //         DAAL_ASSERT(uniquePath[i].partialWeight == 0);
    //     }
    // }

    for (int i = uniqueDepth - 1; i >= 0; --i)
    {
        printf("%p\n", uniquePath + i);

        if (oneFraction != 0)
        {
            const float tmp = nextOnePortion * (uniqueDepth + 1) / static_cast<float>((i + 1) * oneFraction);
            total += tmp;
            nextOnePortion = uniquePath[i].partialWeight - tmp * zeroFraction * ((uniqueDepth - i) / static_cast<float>(uniqueDepth + 1));
        }
        else if (zeroFraction != 0)
        {
            total += (uniquePath[i].partialWeight / zeroFraction) / ((uniqueDepth - i) / static_cast<float>(uniqueDepth + 1));
        }
        else
        {
            DAAL_ASSERT(uniquePath[i].partialWeight == 0);
        }
    }

    return total;
}

} // namespace internal
} // namespace treeshap
} // namespace gbt
} // namespace algorithms
} // namespace daal