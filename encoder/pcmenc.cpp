/*****************************************************************************
**
** Copyright (C) 2006 Arturo Ragozini, Daniel Vik
** Modified by Maxim 2016-2019.
**
**  This software is provided 'as-is', without any express or implied
**  warranty.  In no event will the authors be held liable for any damages
**  arising from the use of this software.
**
**  Permission is granted to anyone to use this software for any purpose,
**  including commercial applications, and to alter it and redistribute it
**  freely, subject to the following restrictions:
**
**  1. The origin of this software must not be misrepresented; you must not
**     claim that you wrote the original software. If you use this software
**     in a product, an acknowledgment in the product documentation would be
**     appreciated but is not required.
**  2. Altered source versions must be plainly marked as such, and must not be
**     misrepresented as being the original software.
**  3. This notice may not be removed or altered from any source distribution.
**
******************************************************************************
* Modifications by Maxim in 2017, 2018
*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iterator>
#include <cstdint>
#include <map>
#include <ctime>
#include <execution>

#include "st.h"
#include "dkm.hpp"
#include "Endian.h"
#include "FileReader.h"
#include "FourCC.h"
#include "Args.h"

// Minimum allowed frequency difference for not doing frequency conversion
constexpr auto minimum_allowed_frequency_difference = 0.005;

enum class PackingType
{
    FourBitRle = 0,
    ThreeBitRle = 1,
    VolByte = 2,
    ChannelVolByte = 3,
    PackedVol = 4,
    Vector6 = 5,
    Vector4 = 6
};

enum class InterpolationType
{
    Linear = 0,
    Quadratic = 1,
    Lagrange11 = 2
};

enum class Chip
{
    // ReSharper disable CppInconsistentNaming
    AY38910 = 0,
    SN76489 = 1
    // ReSharper restore CppInconsistentNaming
};

enum class DataPrecision
{
    Float = 4,
    Double = 8
};

// Resamples a sample from inRate to outRate and returns a new buffer with
// the resampled data and the length of the new buffer.
double* resample(const double* in, const size_t inLen, const unsigned int inRate, const unsigned int outRate, size_t& outLen)
{
    // Configure the resampler
    st_effect_t effect =
    {
        "resample",
        ST_EFF_RATE,
        st_resample_getopts, st_resample_start, st_resample_flow,
        st_resample_drain, st_resample_stop
    };
    st_effect eff{};
    eff.h = &effect;
    st_signalinfo_t iinfo = { (st_rate_t)inRate, 4, 0, 1, Endian::big };
    st_signalinfo_t oinfo = { (st_rate_t)outRate, 4, 0, 1, Endian::big };
    st_updateeffect(&eff, &iinfo, &oinfo, 0);

    // Convert to required format
    const auto ibuf = new st_sample_t[inLen];
    for (size_t i = 0; i < inLen; ++i)
    {
        ibuf[i] = ST_FLOAT_DDWORD_TO_SAMPLE(in[i]);
    }
    const char* argv[] = { "-ql" };
    st_resample_getopts(&eff, 1, argv);
    st_resample_start(&eff);

    // Allocate output buffer
    const uint32_t outBufLen = (uint32_t)((double)inLen * outRate / inRate) + 500;
    const auto obuf = new st_sample_t[outBufLen];

    // Pass samples into resampler
    st_size_t iLen = 0;
    st_size_t oLen = 0;
    for (;;)
    {
        st_size_t idone = ST_BUFSIZ;
        st_size_t odone = ST_BUFSIZ;
        const int rv = st_resample_flow(&eff, ibuf + iLen, obuf + oLen, &idone, &odone);
        iLen += idone;
        oLen += odone;
        if (rv == ST_EOF || iLen + idone > (st_size_t)inLen)
        {
            break;
        }
    }
    delete[] ibuf;

    // Flush resampler
    st_size_t odone = ST_BUFSIZ;
    st_resample_drain(&eff, obuf + oLen, &odone);
    oLen += odone;

    st_resample_stop(&eff);

    // Convert back to double format
    double* outBuf = NULL;
    if (oLen > 0)
    {
        outBuf = new double[oLen];
        for (uint32_t i = 0; i < oLen; ++i)
        {
            outBuf[i] = ST_SAMPLE_TO_FLOAT_DDWORD(obuf[i]);
        }
        outLen = (uint32_t)oLen;
    }
    delete[] obuf;

    return outBuf;
}

// Loads a wav file and creates a new buffer with sample data.
double* loadSamples(const std::string& filename, uint32_t wantedFrequency, size_t& count)
{
    FileReader f(filename);

    f.checkMarker("RIFF");

    const uint32_t riffSize = f.read32();
    if (riffSize != f.size() - 8)
    {
        throw std::runtime_error("File size does not match RIFF header");
    }

    f.checkMarker("WAVE").checkMarker("fmt ");

    uint32_t chunkSize = f.read32();

    const uint16_t formatType = f.read16();
    if (formatType != 0 && formatType != 1)
    {
        throw std::runtime_error("Unsupported format type");
    }

    const uint16_t channels = f.read16();
    if (channels != 1 && channels != 2)
    {
        throw std::runtime_error("Unsupported channel count");
    }

    const uint32_t samplesPerSec = f.read32();

    f.seek(6); // discard avgBytesPerSec (4), blockAlign (2)

    const uint16_t bitsPerSample = f.read16();
    if (bitsPerSample & 0x07)
    {
        throw std::runtime_error("Only supports 8, 16, 24, and 32 bits per sample");
    }

    // Seek to the next chunk
    f.seek(chunkSize - 16);

    while (f.read32() != StringToMarker("data"))
    {
        // Some other chunk
        chunkSize = f.read32();
        f.seek(chunkSize);
    }

    const uint32_t dataSize = f.read32();
    const uint32_t bytesPerSample = ((bitsPerSample + 7) / 8);
    const uint32_t sampleNum = dataSize / bytesPerSample / channels;

    const auto tempSamples = new double[sampleNum];

    for (uint32_t i = 0; i < sampleNum; ++i)
    {
        double value = 0;
        for (int c = 0; c < channels; ++c)
        {
            if (bytesPerSample == 1)
            {
                const uint8_t val = f.read();
                value += ((int)val - 0x80) / 128.0 / channels;
            }
            else {
                uint32_t val = 0;
                for (uint32_t j = 0; j < bytesPerSample; j++)
                {
                    const uint8_t tmp = f.read();
                    val = (val >> 8) | (tmp << 24);
                }
                value += (int)val / 2147483649.0 / channels;
            }
        }
        tempSamples[i] = value;
    }

    double* retSamples;
    if (fabs(1.0 * wantedFrequency / samplesPerSec - 1) < minimum_allowed_frequency_difference)
    {
        retSamples = new double[sampleNum];
        memcpy(retSamples, tempSamples, sampleNum * sizeof(double));
        count = sampleNum;
    }
    else
    {
        printf(" *** WARNING ***\n"
            " Input wave is too far from the target frequency and needs to be resampled.\n"
            " Did you make a mistake with your commandline settings?\n"
            " It's better to resample in a dedicated program for high quality results.\n");
        printf(" Resampling input wave from %dHz to %dHz...", samplesPerSec, wantedFrequency);
        retSamples = resample(tempSamples, sampleNum, samplesPerSec, wantedFrequency, count);
    }

    delete[] tempSamples;

    return retSamples;
}

void dump(const std::string& filename, const uint8_t* pData, size_t byteCount)
{
    std::ofstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    f.open(filename, std::fstream::binary);
    std::copy(pData, pData + byteCount, std::ostream_iterator<uint8_t>(f));
    f.close();
}

/* Lagrange's classical polynomial interpolation */
template <typename T>
static T interpolate(const T* data, int index, T dt, int numLeft, int numRight)
{
    T result = 0.0;
    T t = (T)index + dt;

    for (int j = index - numLeft; j <= index + numRight; ++j)
    {
        T p = data[(j < 0) ? 0 : j];
        for (int k = index - numLeft; k <= index + numRight; ++k)
        {
            if (k != j)
            {
                p *= (t - k) / ((T)j - k);
            }
        }
        result += p;
    }
    return result;
}

// Nasty stuff to get a compile-time-optimised cost function implementation
// (to avoid branching in the inner loop). Because C++ does not allow partial
// function template specialisation, we have to redirect via a templated "impl".
// These should all inline nicely.

// Fallback
template <typename T, unsigned int CostFunction>
struct CostImpl
{
    static T act(T value)
    {
        return pow(fabs(value), CostFunction);
    }
};

// Partial specialisation for n=1
template <typename T>
struct CostImpl<T, 1>
{
    static T calculate(T value)
    {
        return fabs(value);
    }
};

// Partial specialisation for n=2
template <typename T>
struct CostImpl<T, 2>
{
    static T calculate(T value)
    {
        return value * value;
    }
};

// Partial specialisation for n=3
template <typename T>
struct CostImpl<T, 3>
{
    static T calculate(T value)
    {
        return fabs(value * value * value);
    }
};

// Don't bother for higher orders...

// This templated function then calls into the relevant specialised impl
template <typename T, int CostFunction>
T computeCost(T value)
{
    return CostImpl<T, CostFunction>::calculate(value);
}

template <typename T, int CostFunction>
int viterbiInner(
    T* targetOutput, size_t numOutputs,
    T* effectiveVolumesCube,
    uint8_t* precedingValues[256],
    uint8_t* updateValues[256],
    T* dt)
{
    // Costs of previous sample
    T lastCosts[256];
    std::fill_n(lastCosts, sizeof(lastCosts)/sizeof(T), (T)0.0);
    // Costs for each sample
    T sampleCosts[256];
    // These hold some state between each iteration of the loop below...
    unsigned int samplePreceding[256] = {0};
    unsigned int sampleUpdate[256] = {0};

    // For each sample...
    for (size_t t = 0; t < numOutputs; t++)
    {
        // Get the value and channel index
        T sample = targetOutput[t];
        unsigned int channel = t % 3;

        // Initialise our best values to the maximum
        std::fill_n(sampleCosts, 256, std::numeric_limits<T>::max());

        // We print progress every 4K samples
        if (t % 4096 == 0)
        {
            printf("Processing %3.2f%%\r", 100.0 * t / numOutputs);
        }

        T duration = dt[channel];

        // We iterate over the whole "volume cube"...
        for (unsigned int i = 0; i < 16 * 16 * 16; ++i)
        {
            // We can treat i as three indices x, y, z into the volume cube.
            // (It's not stored as a 3D array, maybe for performance?)
            // For each sample, we wan to pick the "best" update to make to our channel
            // for a given pair of values of the other two.
            // This is determined as the cumulative error so far for a given route to the current sample,
            // plus the cost function applied to the deviation in output for a given new value, multiplied by its duration.

            // We transform i to some x, y, z values...
            unsigned int xy = i >> 4;
            unsigned int yz = i & 0xff;

            // We get the value that will be obtained...
            T effectiveVolume = effectiveVolumesCube[i];

            // ...compute the difference between it and what's wanted...
            T deviation = sample - effectiveVolume;

            // ...convert to a cost...
            T cost = duration * computeCost<T, CostFunction>(deviation);

            // ...and add it on to the cumulative cost
            T cumulativeCost = lastCosts[xy] + cost;

            // If it is better than what was computed so far, for a given yz pair, remember it
            // TODO: the result is monotonic (?), we could binary search for it?
            if (cumulativeCost < sampleCosts[yz])
            {
                sampleCosts[yz] = cumulativeCost;
                // And we store the xy and z that go with it
                samplePreceding[yz] = xy;
                sampleUpdate[yz] = i & 0x0f;
            }
        }

        // We now have the lowest-total-cost values for each yz pair.

        // We copy the yz costs as the xy for the next sample
        std::copy(sampleCosts, sampleCosts + 256, lastCosts);

        // And record the other stuff that went with it
        for (int i = 0; i < 256; i++)
        {
            precedingValues[i][t] = (uint8_t)samplePreceding[i];
            updateValues[i][t] = (uint8_t)sampleUpdate[i];
        }
    }

    printf("Processing %3.2f%%\n", 100.0);

    // Now our state arrays contain the final total costs, so we can select the lowest
    auto minIndex = (int)std::distance(lastCosts, std::min_element(lastCosts, lastCosts + 256));

    printf("The cost metric in Viterbi is about %3.3f\n", lastCosts[minIndex]);

    return minIndex;
}

template<typename T>
uint8_t* encode(size_t numOutputs, unsigned int costFunction, T* targetOutput, T* effectiveVolumesCube, T dt[3], bool saveInternal)
{
    // For each of 256 "preceding values" we hold a value per sample
    uint8_t* precedingValues[256];
    // For each of 256 "update values" we hold a value per sample
    uint8_t* updateValues[256];

    // This is the bulk of the memory used: 512 bytes per sample
    for (int i = 0; i < 256; ++i)
    {
        precedingValues[i] = new uint8_t[numOutputs];
        updateValues[i] = new uint8_t[numOutputs];
    }

    printf("   Using cost function: L%d\n", costFunction);

    int minIndex;
    switch (costFunction)
    {
    case 1:
        minIndex = viterbiInner<T, 1>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
        break;
    case 2:
        minIndex = viterbiInner<T, 2>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
        break;
    case 3:
        minIndex = viterbiInner<T, 3>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
        break;
    default:
        throw std::runtime_error("Unhandled cost function >3");
        // Could make a non-templated version of this but I guess it's not needed
        //minIndex = viterbi_inner<T, costFunction>(targetOutput, numOutputs, effectiveVolumesCube, Stt, Itt, dt);
        //break;
    }

    // Then we walk the preceding values and update values for the discovered minimum-cost index
    // backwards to the start
    const auto precedingValuesPath = new uint8_t[numOutputs]; // This is only for the benefit of some analysis below
    const auto updateValuesPath = new uint8_t[numOutputs]; // This is the final result, a series of one-channel updates

    // Set the final values
    precedingValuesPath[numOutputs - 1] = precedingValues[minIndex][numOutputs - 1];
    updateValuesPath[numOutputs - 1] = updateValues[minIndex][numOutputs - 1];
    // And populate backwards
    for (int t = (int)numOutputs - 2; t >= 0; --t)
    {
        // Get the xy values for the sample after this one
        const int xy = precedingValuesPath[t + 1];
        // Obtain the two preceding nibbles for that, so we can iterate backwards
        precedingValuesPath[t] = precedingValues[xy][t];
        // And the z value that goes with them, which is what we really want
        updateValuesPath[t] = updateValues[xy][t];
    }

    // We're done with these now, we copied out the chosen path through them
    for (int i = 0; i < 256; ++i)
    {
        delete[] precedingValues[i];
        delete[] updateValues[i];
    }

    // Then we build a resultant actual-values series by walking the selected path forwards again
    // and building the volume array (i.e. achieved output values)
    auto* achievedOutput = new T[numOutputs];

    for (size_t t = 0; t < numOutputs; ++t)
    {
        int volumeCubeIndex = precedingValuesPath[t] << 4 | updateValuesPath[t];
        achievedOutput[t] = effectiveVolumesCube[volumeCubeIndex];
    }

    if (saveInternal)
    {
        dump("achievedOutput.bin", (uint8_t*)achievedOutput, numOutputs * sizeof(T));
    }

    // Compute the SNR using this (independently of the cost metric used to get it)
    double en = 0;
    double er = 0;
    double mi = 0;
    for (size_t i = 0; i < numOutputs / 3u; i++)
    {
        en += (targetOutput[3u * i + 0u]) * (targetOutput[3u * i + 0u]) * dt[0] +
            (targetOutput[3u * i + 1u]) * (targetOutput[3u * i + 1u]) * dt[1] +
            (targetOutput[3u * i + 2u]) * (targetOutput[3u * i + 2u]) * dt[2];
        er += (targetOutput[3 * i + 0] - achievedOutput[3 * i + 0]) * (targetOutput[3 * i + 0] - achievedOutput[3 * i + 0]) * dt[0] +
            (targetOutput[3 * i + 1] - achievedOutput[3 * i + 1]) * (targetOutput[3 * i + 1] - achievedOutput[3 * i + 1]) * dt[1] +
            (targetOutput[3 * i + 2] - achievedOutput[3 * i + 2]) * (targetOutput[3 * i + 2] - achievedOutput[3 * i + 2]) * dt[2];
        mi += (targetOutput[3 * i + 0]) * dt[0] + (targetOutput[3 * i + 1]) * dt[1] + (targetOutput[3 * i + 2]) * dt[2];
    }

    const double  var = en - mi*mi * 3 / numOutputs;
    printf("SNR is about %3.2f\n", 10 * log10(var / er));

    // We can now delete the data used to compute everything except the final result
    delete[] precedingValuesPath;
    delete[] achievedOutput;

    return updateValuesPath;
}

// Encodes sample data to be played on the PSG.
// The output buffer needs to be three times the size of the input buffer
template <typename T>
uint8_t* encode(
    unsigned int samplesPerTriplet,
    double amplitude,
    const double* samples,
    size_t length,
    unsigned int idt1, unsigned int idt2, unsigned int idt3,
    InterpolationType interpolation,
    unsigned int costFunction,
    bool saveInternal,
    size_t& resultLength,
    const double volumes[16])
{
    const auto start = clock();

    // We normalise the inputs to the range 0..1,
    // plus add some padding on the end to avoid needing range checks at that end
    auto* normalisedInputs = new T[length + 256U];

    const auto minmax = std::minmax_element(samples, samples + length);
    const auto inputMin = *minmax.first;
    const auto inputMax = *minmax.second;
    const auto range = inputMax - inputMin;
    if (range <= 0.0)
    {
        fprintf(stderr, "Error: Sample data is silent\n");
        exit(1);
    }

    for (size_t i = 0u; i < length; i++)
    {
        normalisedInputs[i] = (T)(amplitude * (samples[i] - inputMin) / range);
    }
    std::fill_n(normalisedInputs + length, 256, normalisedInputs[length - 1]);

    // Normalise the relative cycle times to fractions of a triplet time
    T dt[3];
    uint32_t cyclesPerTriplet = idt1 + idt2 + idt3;
    dt[0] = (T)idt1 / cyclesPerTriplet;
    dt[1] = (T)idt2 / cyclesPerTriplet;
    dt[2] = (T)idt3 / cyclesPerTriplet;

    if (samplesPerTriplet < 1)
    {
        samplesPerTriplet = 1;
    }

    printf("Viterbi SNR optimization:\n");
    printf("   %d input samples per PSG triplet output\n", samplesPerTriplet);
    printf("   dt1 = %d  (Normalized: %1.3f)\n", idt1, dt[0]);
    printf("   dt2 = %d  (Normalized: %1.3f)\n", idt2, dt[1]);
    printf("   dt3 = %d  (Normalized: %1.3f)\n", idt3, dt[2]);
    printf("   Using %zu bytes data precision\n", sizeof(T));;

    // Generate a modified version of the inputs to account for any
    // jitter in the output timings, by sampling at the relative offsets
    size_t numOutputs = (length + samplesPerTriplet - 1u) / samplesPerTriplet * 3u;
    auto* targetOutput = new T[numOutputs];

    int numLeft;
    int numRight;

    switch (interpolation)
    {
    case InterpolationType::Linear:
        printf("   Resampling using Linear interpolation...");
        numLeft = 0;
        numRight = 1;
        break;
    case InterpolationType::Quadratic:
        printf("   Resampling using Quadratic interpolation...");
        numLeft = 0;
        numRight = 2;
        break;
    case InterpolationType::Lagrange11:
        printf("   Resampling using Lagrange interpolation on 11 points...");
        numLeft = 5;
        numRight = 5;
        break;
    default:
        throw std::invalid_argument("Invalid interpolation type");
    }

    for (size_t i = 0; i < numOutputs / 3u; i++)
    {
        auto t0 = samplesPerTriplet * i;
        T t1 = (samplesPerTriplet * (i + dt[0]));
        T t2 = (samplesPerTriplet * (i + dt[0] + dt[1]));
        T dt1 = t1 - (int)t1;
        T dt2 = t2 - (int)t2;

        targetOutput[3 * i + 0] = normalisedInputs[t0];
        targetOutput[3 * i + 1] = interpolate(normalisedInputs, (int)t1, dt1, numLeft, numRight);
        targetOutput[3 * i + 2] = interpolate(normalisedInputs, (int)t2, dt2, numLeft, numRight);
    }

    printf(" done (%zu output points)\n", numOutputs);

    if (saveInternal)
    {
        dump("targetOutput.bin", (uint8_t*)targetOutput, numOutputs * sizeof(T));
        dump("normalisedInputs.bin", (uint8_t*)normalisedInputs, (length + 256u) * sizeof(T));
    }

    delete [] normalisedInputs;

    // Build the set of effective volumes for all possible channel settings
    auto effectiveVolumesCube = new T[16 * 16 * 16];
    for (int i = 0; i < 16 * 16 * 16; ++i)
    {
        effectiveVolumesCube[i] = (T)(
            (volumes[(i >> 0) & 0xf] +
            volumes[(i >> 4) & 0xf] +
            volumes[(i >> 8) & 0xf]) / 3.0);
    }

    uint8_t* result = encode(numOutputs, costFunction, targetOutput, effectiveVolumesCube, dt, saveInternal);

    delete[] effectiveVolumesCube;
    delete[] targetOutput;

    const clock_t end = clock();
    const double secondsElapsed = (1.0 * end - start) / CLOCKS_PER_SEC;
    printf(
        "Converted %zu samples to %zu outputs in %.2fs = %.0f samples per second\n",
        length,
        numOutputs,
        secondsElapsed,
        length / secondsElapsed);

    resultLength = numOutputs;
    return result;
}


//////////////////////////////////////////////////////////////////////////////
// RLE encodes a PSG sample buffer. The encoded buffer is created and returned
// by the function.
//
uint8_t* rleEncode(const uint8_t* pData, size_t dataLen, unsigned int rleIncrement, size_t& resultLen)
{
    // Allocate a worst-case-sized buffer
    const auto result = new uint8_t[2 * dataLen + 2];

    // Start with the triplet count
    const size_t tripletCount = dataLen / 3;
    result[0] = (tripletCount >> 0) & 0xff;
    result[1] = (tripletCount >> 8) & 0xff;

    unsigned int currentState[3] = { pData[0], pData[1], pData[2] };
    unsigned int rleCounts[3] = { 0, 0, 0 };
    unsigned int offsets[3] = { 2, 3, 4 };
    unsigned int nextUnusedOffset = 5;

    for (size_t i = 3; i < dataLen; i++)
    {
        const unsigned int channel = i % 3;
        const bool isLastTriplet = i >= dataLen - 3;

        if (currentState[channel] == pData[i] && rleCounts[channel] < 15u - (rleIncrement - 1) && !isLastTriplet)
        {
            rleCounts[channel] += rleIncrement;
        }
        else
        {
            result[offsets[channel]] = (uint8_t)(rleCounts[channel] << 4 | currentState[channel]);
            rleCounts[channel] = 0;
            offsets[channel] = nextUnusedOffset++;
            currentState[channel] = pData[i];
            if (isLastTriplet)
            {
                result[offsets[channel]] = (uint8_t)(rleCounts[channel] << 4 | currentState[channel]);
            }
        }
    }

    resultLen = nextUnusedOffset;

    return result;
}

//////////////////////////////////////////////////////////////////////////////
// Saves an encoded buffer, the file extension is replaced with .bin.
//
void saveEncodedBuffer(const std::string& filename, const uint8_t* buffer, size_t length)
{
    printf("Saving %zu bytes to %s...", length, filename.c_str());
    dump(filename, buffer, length);
    printf("done\n");
}

// Packs data from binBuffer to to destP using the specified packing type
// Consumes only whole triplets
// Consumes at most tripletCount triplets
// Packs <= maxBytes bytes
// Returns the number of triplets consumed - not the number of bytes emitted
size_t chVolPackChunk(uint8_t*& pDest, uint8_t*& pSource, size_t maxTripletCount, size_t maxBytes, PackingType packingType)
{
    // We pack only whole numbers of triplets per bank
    size_t tripletCount;
    switch (packingType)
    {
    case PackingType::VolByte:
    case PackingType::ChannelVolByte:
        tripletCount = std::min(maxTripletCount, (maxBytes - 2) / 3);
        break;
    case PackingType::PackedVol:
        tripletCount = std::min(maxTripletCount, (maxBytes - 2) * 2 / 3);
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }

    if (tripletCount > 0xffff)
    {
        printf("Warning: chunk size %zu truncated\n", tripletCount);
    }

    *pDest++ = (uint8_t)((tripletCount >> 0) & 0xff);
    *pDest++ = (uint8_t)((tripletCount >> 8) & 0xff);

    switch (packingType)
    {
    case PackingType::VolByte:
        std::copy(pSource, pSource + (size_t)tripletCount * 3, pDest);
        pDest += (size_t)tripletCount * 3;
        break;
    case PackingType::ChannelVolByte:
        for (size_t i = 0; i < tripletCount; ++i)
        {
            *pDest++ = (uint8_t)(0 << 6) | pSource[3 * i + 0];
            *pDest++ = (uint8_t)(1 << 6) | pSource[3 * i + 1];
            *pDest++ = (uint8_t)(2 << 6) | pSource[3 * i + 2];
        }
        break;
    case PackingType::PackedVol:
        for (size_t i = 0; i < tripletCount; ++i)
        {
            if (i & 1)
            {
                *(pDest-1) |= pSource[3 * i + 0];
                *pDest++ = (uint8_t)(pSource[3 * i + 1] << 4 | pSource[3 * i + 2] << 0);
            }
            else
            {
                *pDest++ = (uint8_t)(pSource[3 * i + 0] << 4 | pSource[3 * i + 1]);
                *pDest++ = pSource[3 * i + 2] << 4;
            }
        }
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }

    return tripletCount;
}

uint8_t* chVolPack(PackingType packingType, uint8_t* pSource, const size_t sourceLength, const size_t romSplit, size_t& destLength)
{
    auto* result = new uint8_t[2 * sourceLength + 500u];
    uint8_t* pDest = result;
    size_t totalPadding = 0;
    unsigned int bankCount = 0;

    printf("Packing data with ");
    if (romSplit == 0)
    {
        printf("no split");
    }
    else
    {
        printf("splits at %zuKB boundaries", romSplit / 1024);
    }
    switch (packingType)
    {
    case PackingType::VolByte:
        // ReSharper disable once StringLiteralTypo
        printf(", as raw PSG attenuations %%----aaaa\n");
        break;
    case PackingType::ChannelVolByte:
        // ReSharper disable once StringLiteralTypo
        printf(", as channel/attenuation packed bytes %%cc00aaaa\n");
        break;
    case PackingType::PackedVol:
        // ReSharper disable once StringLiteralTypo
        printf(", as packed volume pairs %%aaaabbbb\n");
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }

    if (romSplit == 0)
    {
        chVolPackChunk(pDest, pSource, sourceLength / 3, std::numeric_limits<int>::max(), packingType);
        destLength = pDest - result;
        printf("Packed as %zu bytes of data\n", destLength);
        return result;
    }

    size_t tripletCount = sourceLength / 3;
    while (tripletCount > 0)
    {
        // Pack a bank
        ++bankCount;
        const auto pDestBefore = pDest;
        const size_t tripletsConsumed = chVolPackChunk(pDest, pSource, tripletCount, romSplit, packingType);
        tripletCount -= tripletsConsumed;
        pSource += tripletsConsumed * 3u;
        if (tripletCount > 0)
        {
            // Add padding if needed, but not on the last chunk
            const auto bytesEmitted = pDest - pDestBefore;
            const auto padding = romSplit - bytesEmitted;
            totalPadding += padding;
            for (size_t i = 0; i < padding; ++i)
            {
                *pDest++ = 0;
            }
        }
    };
    destLength = pDest - result;
    printf("Packed as %zu bytes of data (%d banks with %zu bytes padding)\n",
        destLength,
        bankCount,
        totalPadding);
    return result;
}

// RLE encodes a buffer. The method can do both a
// consecutive buffer or a buffer split in multiple buffers
uint8_t* rlePack(uint8_t* binBuffer, size_t length, size_t romSplit, int rleIncrement, size_t& resultLen)
{
    if (romSplit == 0)
    {
        printf("RLE encoding with no split\n");
        const auto result = rleEncode(binBuffer, length, rleIncrement, resultLen);
        printf(
            "- Encoded %zu volume commands (%zu bytes) to %zu bytes of data,\n"
            "  effective compression ratio %.2f%%\n",
            length,
            length / 2,
            resultLen,
            (1.0 * length / 2.0 - resultLen) / (length / 2.0) * 100);
        return result;
    }

    printf("RLE encoding with splits at %zuKB boundaries", romSplit / 1024);

    auto* destBuffer = new uint8_t[2 * (size_t)length];
    uint8_t* pDest = destBuffer;
    resultLen = 0;

    size_t tripletsEncoded = 0;
    size_t tripletsRemaining = length / 3;
    size_t encodedLength;
    size_t totalEncodedLength = 0;
    size_t totalPadding = 0;

    while (tripletsRemaining > 0)
    {
        // We binary search for the point where the packing exceeds the bank size
        size_t tripletCount = std::min(romSplit * 15 / 3, tripletsRemaining); // Starting point: maximum theoretical count (maximum RLE on every sample)
        size_t countLower = 0; // Highest input length which produced a smaller size
        size_t countHigher = std::numeric_limits<unsigned int>::max(); // Lowest input length which produced a larger size
        uint8_t* pEncoded;

        for (;;)
        {
            // Point at the data to compress
            const auto bankSrc = binBuffer + 3 * tripletsEncoded;

            // Compress
            pEncoded = rleEncode(bankSrc, tripletCount * 3, rleIncrement, encodedLength);

            // If it exactly fits, we're done
            if (encodedLength == romSplit)
            {
                break;
            }

            // If we got here, it was no good so we try again
            if (encodedLength > romSplit)
            {
                // If it was bigger, remember that
                countHigher = tripletCount;
            }
            else if (encodedLength < romSplit)
            {
                // If it was smaller, remember that
                countLower = tripletCount;

                // If we are on the last chunk, stop here
                if (tripletCount == tripletsRemaining)
                {
                    countHigher = countLower + 1;
                }
            }

            // If we have found adjacent lengths, we are done
            if (countLower == countHigher - 1)
            {
                if (tripletCount == countHigher)
                {
                    // Need to re-compress
                    delete[] pEncoded;
                    tripletCount = countLower;
                    pEncoded = rleEncode(bankSrc, tripletCount * 3, rleIncrement, encodedLength);
                }
                break;
            }

            // If we don't have a higher point, double
            if (countHigher == std::numeric_limits<int>::max())
            {
                tripletCount *= 2;
            }
            else
            {
                // Else, guess at halfway between them
                tripletCount = (countLower + countHigher) / 2;
            }
        }

        // Update stats
        totalEncodedLength += encodedLength;
        tripletsEncoded += tripletCount;

        // Copy in RLE data
        std::copy(pEncoded, pEncoded + encodedLength, pDest);
        pDest += encodedLength;
        // Blank fill except on the past page
        if (tripletsRemaining > tripletCount)
        {
            const size_t lastPadding = romSplit - encodedLength;
            totalPadding += lastPadding;
            for (size_t i = 0; i < lastPadding; ++i)
            {
                *pDest++ = 0;
            }
        }

        // Show some progress
        printf(".");

        tripletsRemaining -= tripletCount;
    }
    resultLen = (uint32_t)(pDest - destBuffer);
    printf(
        "done\n"
        "- Encoded %zu volume commands (%zu bytes) to %zu bytes of data\n"
        "  (with %zu bytes padding), effective compression ratio %.2f%%\n",
        length,
        length / 2,
        resultLen,
        totalPadding,
        (1.0 * length / 2 - resultLen) / (length / 2.0) * 100);
    return destBuffer;
}

class VectorChunk
{
    uint8_t* const _pDest; // Destination buffer
    const uint8_t* const _pSource;
    size_t _dictionarySize;
    size_t _chunksForThisSplit;
    PackingType _packing;

public:
    VectorChunk(uint8_t* pDest, const uint8_t* pData, size_t dictionarySize, size_t chunksForThisSplit, PackingType packing)
        : _pDest(pDest), _pSource(pData), _dictionarySize(dictionarySize), _chunksForThisSplit(chunksForThisSplit),
          _packing(packing)
    {
    }

private:
    static void dumpPsgAsPcm(const std::string& filename, const uint8_t* pData, size_t size)
    {
        // Convert PSG commands back to a float waveform
        float volumes[16];
        for (int i = 0; i < 15; i++)
        {
            volumes[i] = powf(10.0f, -0.1f * (float)i);
        }
        volumes[15] = 0.0;
        const auto data = new float[size];
        int channels[3] = {};
        for (size_t i = 0; i < size; ++i)
        {
            channels[i % 3] = pData[i];
            data[i] = (volumes[channels[0]] + volumes[channels[1]] + volumes[channels[2]]) / 1.5f - 1.0f;
        }
        // Save it to disk
        ::dump(filename, (const uint8_t*)data, (int)size * sizeof(float));
        delete [] data;
    }

    // Worker method for vector compressing a chunk of data
    // Needs to be templated on the chunk size because of the use of std::array below
    template <size_t N>
    void pack()
    {
        size_t sampleCount = _chunksForThisSplit * N;
        dumpPsgAsPcm("chunk" + std::to_string((unsigned long long)_pSource) + ".bin", _pSource, sampleCount);

        // Convert to the C++ types needed for dkm, also moves the binBuffer pointer on
        // We convert the nibbles to floats, as it needs to maintain an average...
        std::vector<std::array<float, N>> chunks(_chunksForThisSplit);
        const uint8_t* pSource = _pSource;
        for (size_t j = 0; j < _chunksForThisSplit; ++j)
        {
            for (size_t k = 0; k < N; ++k)
            {
                chunks[j][k] = pSource[k];
            }
            pSource += N;
        }

        // Vectorise - this is the majority of the time taken
        auto clusters = dkm::kmeans_lloyd(chunks, 256);

        // Emit the dictionaries
        uint8_t* pDest = _pDest;
        for (const auto& cluster : std::get<0>(clusters))
        {
            uint8_t* p = pDest;
            if constexpr (N % 2 == 0)
            {
                // Even N: we just loop over pairs
                for (auto i = 0U; i < N; i += 2)
                {
                    *p = (uint8_t)(std::lroundf(cluster[i]) << 4 | std::lroundf(cluster[i + 1]));
                    p += 256;
                }
            }
            else
            {
                // Odd N : we need to pack the last one specially
                for (auto i = 0U; i < N - 1; i += 2)
                {
                    *p = (uint8_t)(std::lroundf(cluster[i]) << 4 | std::lroundf(cluster[i + 1]));
                    p += 256;
                }
                *p = (uint8_t)(std::lroundf(cluster[N - 1]) << 4);
            }
            ++pDest;
        }

        // We emit the index count after the dictionary
        pDest = _pDest + _dictionarySize;
        *pDest++ = _chunksForThisSplit & 0xff;
        *pDest++ = (uint8_t)(_chunksForThisSplit >> 8);

        auto indices = std::get<1>(clusters);
        // Followed by the indices
        for (const unsigned& index : indices)
        {
            *pDest++ = (uint8_t)index;
        }

        // Emit samples again, by reconstructing the buffer from the indices
        auto* restored = new uint8_t[sampleCount];
        pDest = restored;
        auto* pVectors = _pDest;
        auto* pIndices = _pDest + _dictionarySize + 2;
        const size_t indexCount = sampleCount / N;
        for (size_t i = 0; i < indexCount; ++i)
        {
            const auto index = *pIndices++;
            // We need to emit N bytes in N/2 chunks
            for (size_t j = 0; j < N/2; ++j)
            {
                const auto pByte = pVectors + j * 256 + index;
                const auto b = *pByte;
                *pDest++ = b >> 4;
                *pDest++ = b & 0xf;
            }
        }
        dumpPsgAsPcm("chunk" + std::to_string((unsigned long long)_pSource) + ".reconstructed.bin", restored, sampleCount);
        delete [] restored;
    }

public:
    void pack()
    {
        switch (_packing)
        {
        case PackingType::Vector4:
            pack<4>();
            break;
        case PackingType::Vector6:
            pack<6>();
            break;
        default:
            throw std::invalid_argument("Invalid packing type");
        }
        printf(".");
    }
};

// Encodes the buffer using vector compression
uint8_t* vectorPack(const PackingType packing, uint8_t* pData, const size_t dataLength, const int romSplit, size_t& destLength)
{
    // Compute the result size
    size_t chunkSize; // Bytes compressed at a time
    size_t dictionarySize; // Size in bytes of the dictionaries for each split

    switch (packing)
    {
    case PackingType::Vector6:
        chunkSize = 6; // 6 nibbles = 3 bytes
        dictionarySize = 256 * 3; // 3 bytes per dictionary entry
        break;
    case PackingType::Vector4:
        chunkSize = 4; // 4 nibbles = 2 bytes
        dictionarySize = 256 * 2; // 2 bytes per dictionary entry
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }
    // Remaining number of output bytes per split
    const size_t outputBytesPerSplit = romSplit - dictionarySize - 2;
    // Number of bytes of input consumed for each total romSplit of output
    const size_t inputBytesPerSplit = outputBytesPerSplit * chunkSize;

    // We need to allocate the result buffer
    // We have as many dictionaries as there are split parts
    const size_t numSplits = dataLength / inputBytesPerSplit + (dataLength % inputBytesPerSplit == 0 ? 0 : 1);
    // And the data is crunched by a factor of the chunk size
    destLength = dataLength / chunkSize + numSplits * (dictionarySize + 2);
    const auto pResult = new uint8_t[destLength];
    auto pDest = pResult; // Working pointer

    printf(
        "Compressing %zu bytes to %zu banks (%zu command dictionary entries), total %zu bytes (%.2f%% compression)",
        dataLength,
        numSplits,
        chunkSize,
        destLength,
        (dataLength - destLength) * 100.0/dataLength);

    size_t chunksRemaining = dataLength / chunkSize; // Truncates! We can't encode partial chunks at EOF

    // We first build a collection of work to do...
    std::vector<VectorChunk> chunks;
    while (chunksRemaining > 0)
    {
        const size_t chunksForThisSplit = std::min(chunksRemaining, outputBytesPerSplit);
        chunksRemaining -= chunksForThisSplit;
        chunks.emplace_back(pDest, pData, dictionarySize, chunksForThisSplit, packing);
        pDest += romSplit;
        pData += chunksForThisSplit * chunkSize;
    }
    // Then we do it in parallel for maximum speed
    std::for_each(
        std::execution::par_unseq,
        chunks.begin(), chunks.end(),
        [](auto&& chunk)
        {
            chunk.pack();
        });
    printf("done\n");
    return pResult;
}

static void skewDown(double* pData, size_t sampleCount, int highPassShift)
{
    // Go forward and generate minimums with smooth return to zero
    std::vector<double> offsets;
    offsets.reserve(sampleCount);
    double min = 0;
    for (double* pSample = pData; pSample < pData + sampleCount; ++pSample)
    {
        double sample = *pData;
        // Find the min of the current value and the low-pass max signal
        min = std::min(min, sample);
        offsets.push_back(min);
        // Make the value decay to 0
        min -= min / (1 << highPassShift);
    }

    // This will have produced steps downwards with slow return to 0,
    // so we go backwards and add smooth returns to 0 in the other direction,
    // and then scale and offset the sample to match
    double* pSample = pData + sampleCount - 1;
    for (int i = (int)sampleCount - 1; i >= 0; --i)
    {
        double sample = *pSample;

        min = std::min(min, sample);
        min = std::min(min, offsets[i]);
        offsets[i] = -min;

        // Then offset the sample
        double adjustedSample = sample - min;
        // Every sample should be >0 now
        assert(adjustedSample >= 0);
        // Then scale down again
        adjustedSample -= 1;

        *pSample-- = adjustedSample;
        min -= min / (1 << highPassShift);
    }
}

// Converts a wav file to PSG binary format, including encoding
void convertWav(const std::string& filename, bool saveInternal, int costFunction, InterpolationType interpolation,
    int cpuFrequency, int dt1, int dt2, int dt3,
    int ratio, double amplitude, int romSplit, PackingType packingType, Chip chip, DataPrecision precision, int smooth)
{
    // Load samples from wav file
    if (ratio < 1)
    {
        throw std::invalid_argument("Invalid number of inputs per output");
    }
    const int frequency = cpuFrequency * ratio / (dt1 + dt2 + dt3);
    if (frequency == 0)
    {
        throw std::invalid_argument("Invalid frequency");
    }

    printf("Encoding PSG samples at %dHz\n", (int)frequency);

    printf("Loading %s...", filename.c_str());
    size_t samplesLen;
    double* samples = loadSamples(filename, frequency, samplesLen);
    if (samples == NULL)
    {
        throw std::runtime_error("Failed to load wav file");
    }
    printf("done; %zu samples\n", samplesLen);

    if (saveInternal)
    {
        dump("samples.bin", (const uint8_t*)samples, samplesLen * sizeof(double));
    }

    if (smooth > 0)
    {
        printf("Skewing samples for better quality...");
        skewDown(samples, samplesLen, smooth);
        printf("done\n");
    }
    if (saveInternal)
    {
        dump("made_positive.bin", (const uint8_t*)samples, samplesLen * sizeof(double));
    }

    // Build the volume table
    double vol[16];
    switch (chip)
    {
    case Chip::AY38910:
        // MSX
        vol[0] = 0;
        for (int i = 1; i < 16; i++)
        {
            vol[i] = pow(2.0, i / 2.0) / pow(2.0, 7.5);
        }
        break;
    case Chip::SN76489:
        // SMS
        for (int i = 0; i < 15; i++)
        {
            vol[i] = pow(10.0, -0.1*i);
        }
        vol[15] = 0.0;
        break;
    default:
        throw std::invalid_argument("Invalid chip");
    }

    // Encode
    size_t binSize;
    uint8_t* binBuffer;
    switch (precision)
    {
    case DataPrecision::Float:
        binBuffer = encode<float>(ratio, amplitude, samples, samplesLen, dt1, dt2, dt3, interpolation, costFunction, saveInternal, binSize, vol);
        break;
    case DataPrecision::Double:
        binBuffer = encode<double>(ratio, amplitude, samples, samplesLen, dt1, dt2, dt3, interpolation, costFunction, saveInternal, binSize, vol);
        break;
    default:
        throw std::invalid_argument("Invalid data precision");
    }

    // Pack
    uint8_t* destBuffer;
    size_t destLength;
    switch (packingType)
    {
    case PackingType::FourBitRle:
        destBuffer = rlePack(binBuffer, binSize, romSplit, 1, destLength);
        break;
    case PackingType::ThreeBitRle:
        destBuffer = rlePack(binBuffer, binSize, romSplit, 2, destLength);
        break;
    case PackingType::VolByte:
    case PackingType::ChannelVolByte:
    case PackingType::PackedVol:
        destBuffer = chVolPack(packingType, binBuffer, binSize, romSplit, destLength);
        break;
    case PackingType::Vector6:
    case PackingType::Vector4:
        destBuffer = vectorPack(packingType, binBuffer, binSize, romSplit, destLength);
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }
    delete[] binBuffer;

    // Save the encoded and packed buffer
    saveEncodedBuffer(filename + ".pcmenc", destBuffer, destLength);
    delete[] destBuffer;
}

int main(int argc, char** argv)
{
    try
    {
        Args args(argc, argv);

        // ReSharper disable StringLiteralTypo
        const auto filename = args.getString("filename", "");
        const auto romSplit = args.getInt("r", 0) * 1024;
        const auto saveInternal = args.exists("si");
        const auto packingType = (PackingType)args.getInt("p", (int)PackingType::FourBitRle);
        const auto ratio = args.getInt("rto", 1);
        const auto interpolation = (InterpolationType)args.getInt("i", (int)InterpolationType::Lagrange11);
        const auto costFunction = args.getInt("c", 2);
        const auto cpuFrequency = args.getInt("cpuf", 3579545);
        const auto amplitude = args.getInt("a", 100);
        const auto dt1 = args.getInt("dt1", 0);
        const auto dt2 = args.getInt("dt2", 0);
        const auto dt3 = args.getInt("dt3", 0);
        const auto chip = (Chip)args.getInt("chip", (int)Chip::SN76489);
        const auto precision = (DataPrecision)args.getInt("precision", (int)DataPrecision::Float);
        const auto smooth = args.getInt("smooth", 0);
        // ReSharper restore StringLiteralTypo

        if (filename.empty())
        {
            // ReSharper disable StringLiteralTypo
            printf(
                "Usage:\n"
                "pcmenc.exe [-r <n>] [-p <packing>] [-cpuf <freq>] \n"
                "           [-dt1 <tstates>] [-dt2 <tstates>] [-dt3 <tstates>]\n"
                "           [-a <amplitude>] [-rto <ratio>] <wavfile>\n"
                "\n"
                "    -r <n>          Pack encoded wave into <n>KB blocks for rom replayers\n"
                "\n"
                "    -p <packing>    Packing type:                b7...b5|b4...b0\n"
                "                        0 = 4bit RLE (default)   run len|PSG vol\n"
                "                        1 = 3 bit RLE; as before but b5 =0\n"
                "                        2 = 1 byte vol\n"
                "                        3 = 1 byte {ch, vol} pairs\n"
                "                        4 = big-endian packed {vol, vol} pairs\n"
                "                        5 = K-means clustered vector tables (6 values per vector)\n"
                "                        6 = K-means clustered vector tables (3 values per vector)\n"
                "\n"
                "    -cpuf <freq>    CPU frequency of the CPU (Hz)\n"
                "                        Default: 3579545\n"
                "\n"
                "    -dt1 <tstates>  CPU Cycles between update of channel A and B\n"
                "    -dt2 <tstates>  CPU Cycles between update of channel B and C\n"
                "    -dt3 <tstates>  CPU Cycles between update of channel C and A\n"
                "                    The replayer sampling base period is \n"
                "                          T = dt1+dt2+dt3\n"
                "                    Note that the replayed sampling base period depends\n"
                "                    on the replayer and how many samples it will play\n"
                "                    in each PSG triplet update.\n"
                "\n"
                "    -smooth <amount>  Low-frequency skewing adjustment decay rate.\n"
                "                        Default 0 = off\n"
                "                        10 is suitable for 44kHz audio\n"
                "\n"
                "    -a <amplitude>  Overdrive amplitude adjustment\n"
                "                        Default 100\n"
                "\n"
                "    -rto <ratio>   Number of input samples per PSG triplet\n"
                "                        Default: 1\n"
                "\n"
                "                   This parameter can be used to oversample the input\n"
                "                   wave. Note that this parameter also will affect the\n"
                "                   replay rate based on how many samples per PSG triplet\n"
                "                   update the replayer uses.\n"
                "\n"
                "    -c <costfun>    Viterbi cost function:\n"
                "                        1  : ABS measure\n"
                "                        2  : Standard MSE (default)\n"
                "                        >2 : Lagrange interpolation of order 'c'\n"
                "\n"
                "    -i <interpol>   Resampling interpolation mode:\n"
                "                        0 = Linear interpolation\n"
                "                        1 = Quadratic interpolation\n"
                "                        2 = Lagrange interpolation (default)\n"
                "\n"
                "    -precision <n>  Main search data precision:\n"
                "                        4 = single precision (default)\n"
                "                        8 = double precision\n"
                "\n"
                "    -chip <chip>    Chip type:\n"
                "                        0 = AY-3-8910/YM2149F (MSX sound chip)\n"
                "                        1 = SN76489/SN76496/NCR8496 (SMS sound chip) (default)\n"
                "\n"
                "    <wavfile>       Filename of .wav file to encode\n"
                "\n");
            // ReSharper restore StringLiteralTypo

            return 0;
        }

        convertWav(filename, saveInternal, costFunction, interpolation, cpuFrequency, dt1, dt2, dt3, ratio, (double)amplitude / 100, romSplit, packingType, chip, precision, smooth);
        return 0;
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
        return 1;
    }
}
