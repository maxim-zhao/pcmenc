/*****************************************************************************
**
** Copyright (C) 2006 Arturo Ragozini, Daniel Vik
** Modified by Maxim 2016-2017.
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
#include <sstream>
#include <map>
#include <ctime>
#include <execution>

#include "st.h"
#include "dkm.hpp"
#include <stdlib.h>

// Minimum allowed frequency difference for not doing frequency conversion
#define MIN_ALLOWED_FREQ_DIFF 0.005

#define MIN(a,b) fmin(a, b) // Beats std::max, same as ?:
#define MAX(a,b) fmax(a, b) // Beats std::max, same as ?:
#define ABS(a)   std::abs(a) // Beats fabs, ?:

#define STR2UL(s) ((uint32_t)(s)[0]<<0|(uint32_t)(s)[1]<<8|(uint32_t)(s)[2]<<16|(uint32_t)(s)[3]<<24)
#define NEED_SWAP() (*(uint32_t*)"A   " == 0x41202020)

enum class PackingType
{
    RLE = 0,
    RLE3 = 1,
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
    AY38910 = 0,
    SN76489 = 1
};

enum class DataPrecision
{
    Float = 4,
    Double = 8
};

// Helper class for file IO
class FileReader
{
    std::ifstream _f;
    uint32_t _size;

public:
    explicit FileReader(const std::string& filename)
    {
        _f.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
        _f.open(filename, std::ifstream::binary);
        _f.seekg(0, std::ios::end);
        _size = (uint32_t)(_f.tellg());
        _f.seekg(0, std::ios::beg);
    }

    ~FileReader()
    {
        _f.close();
    }

    uint32_t read32()
    {
        uint32_t v;
        _f.read((char*)&v, sizeof(uint32_t));
        if (NEED_SWAP())
        {
            v = ((v >> 24) & 0x000000ff) | ((v >> 8) & 0x0000ff00) |
                ((v << 8) & 0x00ff0000) | ((v << 24) & 0xff000000);
        }
        return v;
    }

    uint16_t read16()
    {
        uint16_t v;
        _f.read((char*)&v, sizeof(uint16_t));
        if (NEED_SWAP())
        {
            v = ((v << 8) & 0x00ff0000) | ((v << 24) & 0xff000000);
        }
        return v;
    }

    uint8_t read()
    {
        uint8_t v;
        _f.read((char*)&v, sizeof(uint8_t));
        return v;
    }

    FileReader& checkMarker(const char marker[5])
    {
        if (read32() != STR2UL(marker))
        {
            std::ostringstream ss;
            ss << "Marker " << marker << " not found in file";
            throw std::runtime_error(ss.str());
        }
        return *this;
    }

    uint32_t size() const
    {
        return _size;
    }

    FileReader& seek(const int offset)
    {
        _f.seekg(offset, std::ios::cur);
        return *this;
    }
};

//////////////////////////////////////////////////////////////////////////////
// Resamples a sample from inRate to outRate and returns a new buffer with
// the resampled data and the length of the new buffer.
//
double* resample(const double* in, const int inLen, const int inRate, const int outRate, int& outLen)
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
    st_signalinfo_t iinfo = { (st_rate_t)inRate, 4, 0, 1, NEED_SWAP() };
    st_signalinfo_t oinfo = { (st_rate_t)outRate, 4, 0, 1, NEED_SWAP() };
    st_updateeffect(&eff, &iinfo, &oinfo, 0);

    // Convert to required format
    const auto ibuf = new st_sample_t[inLen];
    for (int i = 0; i < inLen; ++i)
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


//////////////////////////////////////////////////////////////////////////////
// Loads a wav file and creates a new buffer with sample data.
//
double* loadSamples(const std::string& filename, int wantedFrequency, int& count)
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
        throw std::runtime_error("Unsuported format type");
    }

    const uint16_t channels = f.read16();
    if (channels != 1 && channels != 2)
    {
        throw std::runtime_error("Unsuported channel count");
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

    while (f.read32() != STR2UL("data"))
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
    if (ABS(1.0 * wantedFrequency / samplesPerSec - 1) < MIN_ALLOWED_FREQ_DIFF)
    {
        retSamples = new double[sampleNum];
        memcpy(retSamples, tempSamples, sampleNum * sizeof(double));
        count = sampleNum;
    }
    else
    {
        printf(" Resampling input wave from %dHz to %dHz...", (int)samplesPerSec, (int)wantedFrequency);
        retSamples = resample(tempSamples, sampleNum, samplesPerSec, wantedFrequency, count);
    }

    delete[] tempSamples;

    return retSamples;
}

void dump(const std::string& filename, const uint8_t* pData, int byteCount)
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
                p *= (t - k) / (j - k);
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
template <typename T, int costFunction>
struct CostImpl
{
    static T act(T value)
    {
        return pow(fabs(value), costFunction);
    }
};

// Partial specialisation for n=1
template <typename T>
struct CostImpl<T, 1>
{
    static T act(T value)
    {
        return fabs(value);
    }
};

// Partial specialisation for n=2
template <typename T>
struct CostImpl<T, 2>
{
    static T act(T value)
    {
        return value * value;
    }
};

// Partial specialisation for n=3
template <typename T>
struct CostImpl<T, 3>
{
    static T act(T value)
    {
        return fabs(value * value * value);
    }
};

// Don't bother for higher orders...

// This templated function then calls into the relevant specialised impl
template <typename T, int costFunction>
T Cost(T value)
{
    return CostImpl<T, costFunction>::act(value);
}

template <typename T, int costFunction>
int viterbi_inner(T* targetOutput, int numOutputs, T* effectiveVolumesCube, uint8_t* precedingValues[256], uint8_t* updateValues[256], T* dt)
{
    // Costs of previous sample
    T lastCosts[256];
    std::fill_n(lastCosts, _countof(lastCosts), (T)0);
    // Costs for each sample
    T sampleCosts[256];
    // These hold some state between each iteration of the loop below...
    int samplePreceding[256];
    int sampleUpdate[256];

    // For each sample...
    for (int t = 0; t < numOutputs; t++)
    {
        // Get the value and channel index
        T sample = targetOutput[t];
        int channel = t % 3;

        // Initialise our best values to the maximum
        std::fill_n(sampleCosts, 256, std::numeric_limits<T>::max());

        // We print progress every 4K samples
        if (t % 4096 == 0)
        {
            printf("Processing %3.2f%%\r", 100.0 * t / numOutputs);
        }

        // We iterate over the whole "volume cube"...
        for (int i = 0; i < 16 * 16 * 16; ++i)
        {
            // We can treat i as three indices x, y, z into the volume cube.
            // (It's not stored as a 3D array, maybe for performance?)
            // For each sample, we wan to pick the "best" update to make to our channel
            // for a given pair of values of the other two.
            // This is determined as the cumulative error so far for a given route to the current sample,
            // plus the cost function applied to the deviation in output for a given new vale, multiplied by its duration.

            // We transform i to some x, y, z values...
            int xy = i >> 4;
            int yz = i & 0xff;

            // We get the value that will be obtained...
            T effectiveVolume = effectiveVolumesCube[i];

            // ...compute the difference between it and what's wanted...
            T deviation = sample - effectiveVolume;

            // ...convert to a cost...
            T cost = dt[channel] * Cost<T, costFunction>(deviation);

            // ...and add it on to the cumulative cost
            T cumulativeCost = lastCosts[xy] + cost;

            // If it is better than what was computed so far, for a given yz pair, remember it
            // TODO: the result is somewhat linear, we could binary search for it?
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


//////////////////////////////////////////////////////////////////////////////
// Encodes sample data to be played on the PSG.
// The output buffer needs to be three times the size of the input buffer
//
template <typename T>
uint8_t* encode(int samplesPerTriplet, double amplitude, const double* samples, int length,
    int idt1, int idt2, int idt3,
    InterpolationType interpolation, int costFunction,
    bool saveInternal, int& resultLength, const double volumes[16])
{
    const clock_t start = clock();

    // We normalise the inputs to the range 0..1, 
    // plus add some padding on the end to avoid needing range checks at that end
    auto* normalisedInputs = new T[length + 256];

    const auto minmax = std::minmax_element(samples, samples + length);
    const auto inputMin = *minmax.first;
    const auto inputMax = *minmax.second;

    for (int i = 0; i < length; i++)
    {
        normalisedInputs[i] = (T)(amplitude * (samples[i] - inputMin) / (inputMax - inputMin));
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
    int numOutputs = (length + samplesPerTriplet - 1) / samplesPerTriplet * 3;
    auto* targetOutput = new T[numOutputs];

    int numLeft;
    int numRight;

    switch (interpolation)
    {
    case InterpolationType::Linear:
        printf("   Resampling using Linear interpolation\n");
        numLeft = 0;
        numRight = 1;
        break;
    case InterpolationType::Quadratic:
        printf("   Resampling using Quadratic interpolation\n");
        numLeft = 0;
        numRight = 2;
        break;
    case InterpolationType::Lagrange11:
        printf("   Resampling using Lagrange interpolation on 11 points\n");
        numLeft = 5;
        numRight = 5;
        break;
    default:
        throw std::invalid_argument("Invalid interpolation type");
    }

    for (int i = 0; i < numOutputs / 3; i++)
    {
        int t0 = (samplesPerTriplet * i);
        T t1 = (samplesPerTriplet * (i + dt[0]));
        T t2 = (samplesPerTriplet * (i + dt[0] + dt[1]));
        T dt1 = t1 - (int)t1;
        T dt2 = t2 - (int)t2;

        targetOutput[3 * i + 0] = normalisedInputs[t0];
        targetOutput[3 * i + 1] = interpolate(normalisedInputs, (int)t1, dt1, numLeft, numRight);
        targetOutput[3 * i + 2] = interpolate(normalisedInputs, (int)t2, dt2, numLeft, numRight);
    }

    if (saveInternal)
    {
        dump("targetOutput.bin", (uint8_t*)targetOutput, numOutputs * sizeof(T));
        dump("normalisedInputs.bin", (uint8_t*)normalisedInputs, (length * 256) * sizeof(T));
    }

    // Build the set of effective volumes for all possible channel settings
    T effectiveVolumesCube[16 * 16 * 16];
    for (int i = 0; i < 16 * 16 * 16; ++i)
    {
        effectiveVolumesCube[i] = (T)(
            volumes[(i >> 0) & 0xf] +
            volumes[(i >> 4) & 0xf] +
            volumes[(i >> 8) & 0xf]);
    }

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
        minIndex = viterbi_inner<T, 1>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
        break;
    case 2:
        minIndex = viterbi_inner<T, 2>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
        break;
    case 3:
        minIndex = viterbi_inner<T, 3>(targetOutput, numOutputs, effectiveVolumesCube, precedingValues, updateValues, dt);
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
    for (int t = numOutputs - 2; t >= 0; --t)
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

    for (int t = 0; t < numOutputs; ++t)
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
    for (int i = 0; i < numOutputs / 3; i++)
    {
        en += (targetOutput[3 * i + 0]) * (targetOutput[3 * i + 0]) * dt[0] +
            (targetOutput[3 * i + 1]) * (targetOutput[3 * i + 1]) * dt[1] +
            (targetOutput[3 * i + 2]) * (targetOutput[3 * i + 2]) * dt[2];
        er += (targetOutput[3 * i + 0] - achievedOutput[3 * i + 0]) * (targetOutput[3 * i + 0] - achievedOutput[3 * i + 0]) * dt[0] +
            (targetOutput[3 * i + 1] - achievedOutput[3 * i + 1]) * (targetOutput[3 * i + 1] - achievedOutput[3 * i + 1]) * dt[1] +
            (targetOutput[3 * i + 2] - achievedOutput[3 * i + 2]) * (targetOutput[3 * i + 2] - achievedOutput[3 * i + 2]) * dt[2];
        mi += (targetOutput[3 * i + 0]) * dt[0] + (targetOutput[3 * i + 1]) * dt[1] + (targetOutput[3 * i + 2]) * dt[2];
    }

    const double  var = en - mi*mi * 3 / numOutputs;
    printf("SNR is about %3.2f\n", 10 * log10(var / er));

    // We can now delete the data used to compute everything except the final result
    delete[] normalisedInputs;
    delete[] targetOutput;
    delete[] precedingValuesPath;
    delete[] achievedOutput;

    const clock_t end = clock();
    const double secondsElapsed = (1.0 * end - start) / CLOCKS_PER_SEC;
    printf(
        "Converted %d samples to %d outputs in %.2fs = %.0f samples per second\n",
        length,
        numOutputs,
        secondsElapsed,
        length / secondsElapsed);

    resultLength = numOutputs;
    return updateValuesPath;
}


//////////////////////////////////////////////////////////////////////////////
// RLE encodes a PSG sample buffer. The encoded buffer is created and returned
// by the function.
//
uint8_t* rleEncode(const uint8_t* pData, int dataLen, int rleIncrement, int& resultLen)
{
    // Allocate a worst-case-sized buffer
    const auto result = new uint8_t[2 * dataLen + 2];

    // Start with the triplet count
    const size_t tripletCount = dataLen / 3;
    result[0] = (tripletCount >> 0) & 0xff;
    result[1] = (tripletCount >> 8) & 0xff;

    int currentState[3] = { pData[0], pData[1], pData[2] };
    int rleCounts[3] = { 0, 0, 0 };
    int offsets[3] = { 2, 3, 4 };
    int nextUnusedOffset = 5;

    for (int i = 3; i < dataLen; i++)
    {
        const int channel = i % 3;
        const bool isLastTriplet = i >= dataLen - 3;

        if (currentState[channel] == pData[i] && rleCounts[channel] < 15 - (rleIncrement - 1) && !isLastTriplet)
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
void saveEncodedBuffer(const std::string& filename, const uint8_t* buffer, int length)
{
    printf("Saving %d bytes to %s...", length, filename.c_str());
    dump(filename, buffer, length);
    printf("done\n");
}

// Packs data from binBuffer to to destP using the specified packing type
// Consumes only whole triplets
// Consumes at most tripletCount triplets
// Packs <= maxBytes bytes
// Returns the number of triplets consumed - not the number of bytes emitted
int chVolPackChunk(uint8_t*& pDest, uint8_t*& pSource, int maxTripletCount, int maxBytes, PackingType packingType)
{
    // We pack only whole numbers of triplets per bank
    int tripletCount;
    switch (packingType)
    {
    case PackingType::VolByte:
    case PackingType::ChannelVolByte:
        tripletCount = std::min(maxTripletCount, (maxBytes - 2) / 3);
        break;
    case PackingType::PackedVol:
        tripletCount = std::min(maxTripletCount, (int)(((int64_t)maxBytes - 2) * 2 / 3));
        break;
    default:
        throw std::invalid_argument("Invalid packing type");
    }

    if (tripletCount > 0xffff)
    {
        printf("Warning: chunk size %d truncated\n", tripletCount);
    }

    *pDest++ = (uint8_t)((tripletCount >> 0) & 0xff);
    *pDest++ = (uint8_t)((tripletCount >> 8) & 0xff);

    switch (packingType)
    {
    case PackingType::VolByte:
        std::copy(pSource, pSource + tripletCount * 3, pDest);
        pDest += tripletCount * 3;
        break;
    case PackingType::ChannelVolByte:
        for (int i = 0; i < tripletCount; ++i)
        {
            *pDest++ = (uint8_t)(0 << 6) | pSource[3 * i + 0];
            *pDest++ = (uint8_t)(1 << 6) | pSource[3 * i + 1];
            *pDest++ = (uint8_t)(2 << 6) | pSource[3 * i + 2];
        }
        break;
    case PackingType::PackedVol:
        for (int i = 0; i < tripletCount; ++i)
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

uint8_t* chVolPack(PackingType packingType, uint8_t* pSource, const int sourceLength, const int romSplit, int& destLength)
{
    auto* result = new uint8_t[2 * sourceLength + 500];
    uint8_t* pDest = result;
    int totalPadding = 0;
    int bankCount = 0;
    if (romSplit == 0)
    {
        chVolPackChunk(pDest, pSource, sourceLength / 3, std::numeric_limits<int>::max(), packingType);
    }
    else
    {
        int tripletCount = sourceLength / 3;
        while (tripletCount > 0)
        {
            // Pack a bank
            ++bankCount;
            const auto pDestBefore = pDest;
            const int tripletsConsumed = chVolPackChunk(pDest, pSource, tripletCount, romSplit, packingType);
            tripletCount -= tripletsConsumed;
            pSource += tripletsConsumed * 3;
            if (tripletCount > 0)
            {
                // Add padding if needed, but not on the last chunk
                const auto bytesEmitted = (int)(pDest - pDestBefore);
                const int padding = romSplit - bytesEmitted;
                totalPadding += padding;
                for (int i = 0; i < padding; ++i)
                {
                    *pDest++ = 0;
                }
            }
        };
    }
    destLength = (uint32_t)(pDest - result);
    printf("Saved as %d bytes of data (%d banks with %d bytes padding)\n",
        destLength,
        bankCount,
        totalPadding);
    return result;
}

// RLE encodes a buffer. The method can do both a
// consecutive buffer or a buffer split in multiple buffers
uint8_t* rlePack(uint8_t* binBuffer, uint32_t length, int romSplit, int rleIncrement, int& resultLen)
{
    if (romSplit == 0)
    {
        printf("RLE encoding with no split\n");
        const auto result = rleEncode(binBuffer, length, rleIncrement, resultLen);
        printf(
            "- Encoded %d volume commands (%d bytes) to %d bytes of data,\n"
            "  effective compression ratio %.2f%%\n",
            length,
            length / 2,
            resultLen,
            (1.0 * length / 2.0 - resultLen) / (length / 2.0) * 100);
        return result;
    }

    printf("RLE encoding with splits at %dKB boundaries", romSplit / 1024);

    auto* destBuffer = new uint8_t[2 * length];
    uint8_t* pDest = destBuffer;
    resultLen = 0;

    int tripletsEncoded = 0;
    int tripletsRemaining = length / 3;
    int encodedLength;
    int totalEncodedLength = 0;
    int totalPadding = 0;

    while (tripletsRemaining > 0)
    {
        // We binary search for the point where the packing exceeds the bank size
        int tripletCount = std::min(romSplit * 15 / 3, tripletsRemaining); // Starting point: maximum theoretical count (maximum RLE on every sample)
        int countLower = 0; // Highest input length which produced a smaller size
        int countHigher = std::numeric_limits<int>::max(); // Lowest input length whch produced a larger size
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
            const int lastPadding = romSplit - encodedLength;
            totalPadding += lastPadding;
            for (int i = 0; i < lastPadding; ++i)
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
        "- Encoded %d volume commands (%d bytes) to %d bytes of data\n"
        "  (with %d bytes padding), effective compression ratio %.2f%%\n",
        length,
        length / 2,
        resultLen,
        totalPadding,
        (1.0 * length / 2 - resultLen) / (length / 2.0) * 100);
    return destBuffer;
}

class VectorChunk
{
    uint8_t* _pDest;
    const uint8_t* _pSource;
    int _dictionarySize;
    int _chunksForThisSplit;
    PackingType _packing;

public:
    VectorChunk(uint8_t* pDest, const uint8_t* pData, int dictionarySize, int chunksForThisSplit, PackingType packing)
        : _pDest(pDest), _pSource(pData), _dictionarySize(dictionarySize), _chunksForThisSplit(chunksForThisSplit),
          _packing(packing)
    {
    }

private:
    // Worker method for vector compressing a chunk of data
    // Needs to be templated on the chunk size because of the use of std::array below
    template <size_t N>
    void pack()
    {
        // Convert to the C++ types needed for dkm, also moves the binBuffer pointer on
        // We convert the nibbles to floats, as it needs to maintain an average...
        std::vector<std::array<float, N>> chunks(_chunksForThisSplit);
        for (auto j = 0; j < _chunksForThisSplit; ++j)
        {
            for (auto k = 0U; k < N; ++k)
            {
                chunks[j][k] = _pSource[k];
            }
            _pSource += N;
        }

        // Vectorise - this is the majority of the time taken
        auto clusters = dkm::kmeans_lloyd(chunks, 256);

        // Emit the dictionaries
        for (const auto& cluster : std::get<0>(clusters))
        {
            auto p = _pDest;
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
            ++_pDest;
        }
        //The pointer is left at the end of the first dictionary, we move it past the rest
        _pDest += _dictionarySize - 256;
        // And the index count
        *_pDest++ = _chunksForThisSplit & 0xff;
        *_pDest++ = (uint8_t)(_chunksForThisSplit >> 8);

        auto indices = std::get<1>(clusters);
        // Emit the samples
        for (const unsigned& index : indices)
        {
            *_pDest++ = (uint8_t)index;
        }
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
uint8_t* vectorPack(const PackingType packing, uint8_t* pData, const int dataLength, const int romSplit, int& destLength)
{
    // Compute the result size
    int chunkSize; // Bytes compressed at a time
    int dictionarySize; // Size in bytes of the dictionaries for each split

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
    const int outputBytesPerSplit = romSplit - dictionarySize - 2;
    // Number of bytes of input consumed for each total romSplit of output
    const int inputBytesPerSplit = outputBytesPerSplit * chunkSize;

    // We need to allocate the result buffer
    // We have as many dictionaries as there are split parts
    const int numSplits = dataLength / inputBytesPerSplit + (dataLength % inputBytesPerSplit == 0 ? 0 : 1);
    // And the data is crunched by a factor of the chunk size
    destLength = dataLength / chunkSize + numSplits * (dictionarySize + 2);
    const auto pResult = new uint8_t[destLength];
    auto pDest = pResult; // Working pointer

    printf(
        "Compressing %d bytes to %d banks (%d command dictionary entries), total %d bytes (%.2f%% compression)",
        dataLength,
        numSplits,
        chunkSize,
        destLength,
        (dataLength - destLength) * 100.0/dataLength);

    int chunksRemaining = dataLength / chunkSize; // Truncates! We can't encode partial chunks at EOF

    // We first build a collection of work to do...
    std::vector<VectorChunk> chunks;
    while (chunksRemaining > 0)
    {
        const int chunksForThisSplit = std::min(chunksRemaining, outputBytesPerSplit);
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

// Converts a wav file to PSG binary format, including encoding
void convertWav(const std::string& filename, bool saveInternal, int costFunction, InterpolationType interpolation,
    int cpuFrequency, int dt1, int dt2, int dt3,
    int ratio, double amplitude, int romSplit, PackingType packingType, Chip chip, DataPrecision precision)
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
    int samplesLen;
    double* samples = loadSamples(filename, frequency, samplesLen);
    if (samples == NULL)
    {
        throw std::runtime_error("Failed to load wav file");
    }
    printf("done\n");

    // Do viterbi encoding
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

    int binSize;
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

    // RLE encode the buffer. Either as one consecutive RLE encoded
    // buffer, or as 8kB small buffers, each RLE encoded with header.
    uint8_t* destBuffer;
    int destLength;

    switch (packingType)
    {
    case PackingType::RLE:
        destBuffer = rlePack(binBuffer, binSize, romSplit, 1, destLength);
        break;
    case PackingType::RLE3:
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

    // Save the encoded buffer
    saveEncodedBuffer(filename + ".pcmenc", destBuffer, destLength);
    delete[] destBuffer;
}

// A class which parses the commandline args into an internal dictionary, and then does some type conversion for us.
class Args
{
    std::map<std::string, std::string> _args;

public:
    Args(int argc, char** argv)
    {
        bool haveLastKey = false;
        std::map<std::string, std::string>::iterator lastKey;
        for (int i = 1; i < argc; ++i)
        {
            switch (argv[i][0])
            {
            case '/':
            case '-':
                // Store as a valueless key
                lastKey = _args.insert(make_pair(std::string(argv[i] + 1), "")).first;
                haveLastKey = true;
                // Remember it
                break;
            case '\0':
                break;
            default:
                // Must be a value for the last key, or a filename
                if (haveLastKey)
                {
                    lastKey->second = argv[i];
                }
                else
                {
                    _args.insert(std::make_pair("filename", argv[i]));
                }
                // Clear it so we don't put the filename in the wrong place
                haveLastKey = false;
                break;
            }
        }
    }

    std::string getString(const std::string& name, const std::string& defaultValue)
    {
        const auto it = _args.find(name);
        if (it == _args.end())
        {
            return defaultValue;
        }
        return it->second;
    }

    int getInt(const std::string& name, uint32_t defaultValue)
    {
        const auto it = _args.find(name);
        if (it == _args.end())
        {
            return defaultValue;
        }
        return std::strtol(it->second.c_str(), nullptr, 10);
    }

    bool exists(const std::string& name) const
    {
        return _args.find(name) != _args.end();
    }
};


int main(int argc, char** argv)
{
    try
    {
        Args args(argc, argv);

        auto filename = args.getString("filename", "");
        const auto romSplit = args.getInt("r", 0) * 1024;
        const auto saveInternal = args.exists("si");
        const auto packingType = (PackingType)args.getInt("p", (int)PackingType::RLE);
        const auto ratio = args.getInt("rto", 1);
        const auto interpolation = (InterpolationType)args.getInt("i", (int)InterpolationType::Lagrange11);
        const auto costFunction = args.getInt("c", 2);
        const auto cpuFrequency = args.getInt("cpuf", 3579545);
        const auto amplitude = args.getInt("a", 115);
        const auto dt1 = args.getInt("dt1", 0);
        const auto dt2 = args.getInt("dt2", 0);
        const auto dt3 = args.getInt("dt3", 0);
        const auto chip = (Chip)args.getInt("chip", (int)Chip::SN76489);
        const auto precision = (DataPrecision)args.getInt("precision", (int)DataPrecision::Float);

        if (filename.empty())
        {
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
                "    -a <amplitude>  Overdrive amplitude adjustment\n"
                "                        Default 115\n"
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

            return 0;
        }

        convertWav(filename, saveInternal, costFunction, interpolation, cpuFrequency, dt1, dt2, dt3, ratio, (double)amplitude / 100, romSplit, packingType, chip, precision);
        return 1;
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
        return 0;
    }
}
