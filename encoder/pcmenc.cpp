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
*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iterator>
#include <cstdint>
#include <sstream>
#include <map>
#include <ctime>

#include "st.h"

// Minimum allowed frequency difference for not doing frequency conversion
#define MIN_ALLOWED_FREQ_DIFF 0.005

#define MIN(a,b) fmin(a, b) // Beats std::max, same as ?:
#define MAX(a,b) fmax(a, b) // Beats std::max, same as ?:
#define ABS(a)   std::abs(a) // Beats fabs, ?:

#define str2ul(s) ((uint32_t)s[0]<<0|(uint32_t)s[1]<<8|(uint32_t)s[2]<<16|(uint32_t)s[3]<<24)
#define needSwap() (*(uint32_t*)"A   " == 0x41202020)

enum PackingType
{
	PackingType_RLE = 0,
	PackingType_RLE3 = 1,
	PackingType_VolByte = 2,
	PackingType_ChannelVolByte = 3,
	PackingType_PackedVol = 4
};

enum InterpolationType
{
	Interpolation_Linear = 0,
	Interpolation_Quadratic = 1,
	Interpolation_Lagrange11 = 2
};

enum Chip
{
	Chip_AY38910 = 0,
	Chip_SN76489 = 1
};

enum DataPrecision
{
	DataPrecision_Float = 4,
	DataPrecision_Double = 8
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
		if (needSwap())
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
		if (needSwap())
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

	FileReader& checkMarker(char marker[5])
	{
		if (read32() != str2ul(marker))
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

	FileReader& seek(int offset)
	{
		_f.seekg(offset, std::ios::cur);
		return *this;
	}
};

//////////////////////////////////////////////////////////////////////////////
// Resamples a sample from inRate to outRate and returns a new buffer with
// the resampled data and the length of the new buffer.
//
double* resample(double* in, int inLen, int inRate, int outRate, int& outLen)
{
	// Configure the resampler
	st_effect_t effect =
	{
		"resample",
		ST_EFF_RATE,
		st_resample_getopts, st_resample_start, st_resample_flow,
		st_resample_drain, st_resample_stop
	};
	st_effect eff;
	eff.h = &effect;
	st_signalinfo_t iinfo = { (st_rate_t)inRate, 4, 0, 1, needSwap() };
	st_signalinfo_t oinfo = { (st_rate_t)outRate, 4, 0, 1, needSwap() };
	st_updateeffect(&eff, &iinfo, &oinfo, 0);

	// Convert to required format
	st_sample_t* ibuf = new st_sample_t[inLen];
	for (int i = 0; i < inLen; ++i)
	{
		ibuf[i] = ST_FLOAT_DDWORD_TO_SAMPLE(in[i]);
	}
	char* argv[] = { "-ql" };
	st_resample_getopts(&eff, 1, argv);
	st_resample_start(&eff);

	// Allocate output buffer
	uint32_t outBufLen = (uint32_t)((double)inLen * outRate / inRate) + 500;
	st_sample_t* obuf = new st_sample_t[outBufLen];

	// Pass samples into resampler
	st_size_t iLen = 0;
	st_size_t oLen = 0;
	for (;;)
	{
		st_size_t idone = ST_BUFSIZ;
		st_size_t odone = ST_BUFSIZ;
		int rv = st_resample_flow(&eff, ibuf + iLen, obuf + oLen, &idone, &odone);
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

	uint32_t riffSize = f.read32();
	if (riffSize != f.size() - 8)
	{
		throw std::runtime_error("File size does not match RIFF header");
	}

	f.checkMarker("WAVE").checkMarker("fmt ");

	uint32_t chunkSize = f.read32();

	uint16_t formatType = f.read16();
	if (formatType != 0 && formatType != 1)
	{
		throw std::runtime_error("Unsuported format type");
	}

	uint16_t channels = f.read16();
	if (channels != 1 && channels != 2)
	{
		throw std::runtime_error("Unsuported channel count");
	}

	uint32_t samplesPerSec = f.read32();

	f.seek(6); // discard avgBytesPerSec (4), blockAlign (2)

	uint16_t bitsPerSample = f.read16();
	if (bitsPerSample & 0x07)
	{
		throw std::runtime_error("Only supports 8, 16, 24, and 32 bits per sample");
	}

	// Seek to the next chunk
	f.seek(chunkSize - 16);

	while (f.read32() != str2ul("data"))
	{
		// Some other chunk
		chunkSize = f.read32();
		f.seek(chunkSize);
	}

	uint32_t dataSize = f.read32();
	uint32_t bytesPerSample = ((bitsPerSample + 7) / 8);
	uint32_t sampleNum = dataSize / bytesPerSample / channels;

	double* tempSamples = new double[sampleNum];

	for (uint32_t i = 0; i < sampleNum; ++i)
	{
		double value = 0;
		for (int c = 0; c < channels; ++c)
		{
			if (bytesPerSample == 1)
			{
				uint8_t val = f.read();
				value += ((int)val - 0x80) / 128.0 / channels;
			}
			else {
				uint32_t val = 0;
				for (uint32_t j = 0; j < bytesPerSample; j++)
				{
					uint8_t tmp = f.read();
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
		printf("Resampling input wave from %dHz to %dHz\n", (int)samplesPerSec, (int)wantedFrequency);
		retSamples = resample(tempSamples, sampleNum, samplesPerSec, wantedFrequency, count);
	}

	delete[] tempSamples;

	return retSamples;
}

void dump(const std::string filename, const uint8_t* pData, int byteCount)
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
	std::fill_n(lastCosts, 256, (T)0);
	// Starting point for min-cost search for each target
	T maxCosts[256];
	std::fill_n(maxCosts, 256, std::numeric_limits<T>::max());
	int samplePreceding[256];
	int sampleUpdate[256];

	for (int t = 0; t < numOutputs; t++)
	{
		T sample = targetOutput[t];
		int channel = t % 3;

		// For each preceding two 4-bit values, we calculate the minimum cost for this sample
		T sampleCosts[256];
		std::copy(maxCosts, maxCosts + 256, sampleCosts);

		if (t % 4096 == 0)
		{
			printf("Processing %3.2f%%\r", 100.0 * t / numOutputs);
		}

		for (int i = 0; i < 16 * 16 * 16; ++i)
		{
			T effectiveVolume = effectiveVolumesCube[i];

			// We can imagine the index i as actually three channel indexes xyz on the cube
			// We are going to judge the quality of z assuming the other two channels are x and y
			int xy = i >> 4;

			// This is the delta between what we might achieve and what we want
			T normVal = sample - effectiveVolume;
			// This is our judgement of how bad that is, plus the cost
			// of the previous sample (assuming it lead to xy)
			T cost = lastCosts[xy] + dt[channel] * Cost<T, costFunction>(normVal);

			// We select the minimum costs for each yz
			int yz = i & 0xff;
			if (cost < sampleCosts[yz])
			{
				sampleCosts[yz] = cost;
				// And we store the xy and z that go with it
				samplePreceding[yz] = xy;
				sampleUpdate[yz] = i & 0x0f;
			}
		}

		// Then we copy the yz costs as the xy for the next sample
		std::copy(sampleCosts, sampleCosts + 256, lastCosts);

		// And record the other stuff that went with it
		for (int i = 0; i < 256; i++)
		{
			precedingValues[i][t] = (uint8_t)samplePreceding[i];
			updateValues[i][t] = (uint8_t)sampleUpdate[i];
		}
	}

	printf("Processing %3.2f%%\n", 100.0);

	// We select the smallest total cost
	int minIndex = (int)std::distance(lastCosts, std::min_element(lastCosts, lastCosts + 256));

	printf("The cost metric in Viterbi is about %3.3f\n", lastCosts[minIndex]);

	return minIndex;
}


//////////////////////////////////////////////////////////////////////////////
// Encodes sample data to be played on the PSG.
// The output buffer needs to be three times the size of the input buffer
//
template <typename T>
uint8_t* viterbi(int samplesPerTriplet, double amplitude, const double* samples, int length,
	int idt1, int idt2, int idt3,
	InterpolationType interpolation, int costFunction,
	bool saveInternal, int& binSize, const double vol[16])
{
	clock_t start = clock();

	// We normalise the inputs to the range 0..1, 
	// plus add some padding on the end to avoid needing range checks at that end
	T* normalisedInputs = new T[length + 256];

	auto minmax = std::minmax_element(samples, samples + length);
	auto inputMin = *minmax.first;
	auto inputMax = *minmax.second;

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
	printf("   dt1 = %d  (Normalized: %1.3f)\n", (int)idt1, dt[0]);
	printf("   dt2 = %d  (Normalized: %1.3f)\n", (int)idt2, dt[1]);
	printf("   dt3 = %d  (Normalized: %1.3f)\n", (int)idt3, dt[2]);
	printf("   Using %llu bytes data precision\n", sizeof(T));;

	// Generate a modified version of the inputs to account for any
	// jitter in the output timings, by sampling at the relative offsets
	int numOutputs = (length + samplesPerTriplet - 1) / samplesPerTriplet * 3;
	T* targetOutput = new T[numOutputs];

	int numLeft;
	int numRight;

	switch (interpolation)
	{
	case Interpolation_Linear:
		printf("   Resampling using Linear interpolation\n");
		numLeft = 0;
		numRight = 1;
		break;
	case Interpolation_Quadratic:
		printf("   Resampling using Quadratic interpolation\n");
		numLeft = 0;
		numRight = 2;
		break;
	case Interpolation_Lagrange11:
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
			vol[(i >> 0) & 0xf] +
			vol[(i >> 4) & 0xf] +
			vol[(i >> 8) & 0xf]);
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
	uint8_t* precedingValuesPath = new uint8_t[numOutputs]; // This is only for the benefit of some analysis below
	uint8_t* updateValuesPath = new uint8_t[numOutputs]; // This is the final result, a series of one-channel updates

	precedingValuesPath[numOutputs - 1] = precedingValues[minIndex][numOutputs - 1];
	updateValuesPath[numOutputs - 1] = updateValues[minIndex][numOutputs - 1];
	for (int t = numOutputs - 2; t >= 0; --t)
	{
		int xy = precedingValuesPath[t + 1];
		precedingValuesPath[t] = precedingValues[xy][t];
		updateValuesPath[t] = updateValues[xy][t];
	}

	// We're done with these now, we copied out the chosen path through them
	for (int i = 0; i < 256; ++i)
	{
		delete[] precedingValues[i];
		delete[] updateValues[i];
	}

	// Then we build a resultant actual-values series by walking the selected path forwards again
	T* achievedOutput = new T[numOutputs];

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

	double  var = en - mi*mi * 3 / numOutputs;
	printf("SNR is about %3.2f\n", 10 * log10(var / er));

	delete[] normalisedInputs;
	delete[] targetOutput;
	delete[] precedingValuesPath;
	delete[] achievedOutput;

	clock_t end = clock();
	double secondsElapsed = (1.0 * end - start) / CLOCKS_PER_SEC;
	printf(
		"Converted %d samples to %d outputs in %.2fs = %.0f samples per second\n",
		length,
		numOutputs,
		secondsElapsed,
		length / secondsElapsed);

	binSize = numOutputs;
	return updateValuesPath;
}


//////////////////////////////////////////////////////////////////////////////
// RLE encodes a PSG sample buffer. The encoded buffer is created and returned
// by the function.
//
uint8_t* rleEncode(const uint8_t* pData, int dataLen, int rleIncrement, int& resultLen)
{
	// Allocate a worst-case-sized buffer
	uint8_t* result = new uint8_t[2 * dataLen + 2];

	// Start with the triplet count
	size_t tripletCount = dataLen / 3;
	result[0] = (tripletCount >> 0) & 0xff;
	result[1] = (tripletCount >> 8) & 0xff;

	int currentState[3] = { pData[0], pData[1], pData[2] };
	int rleCounts[3] = { 0, 0, 0 };
	int offsets[3] = { 2, 3, 4 };
	int nextUnusedOffset = 5;

	for (int i = 3; i < dataLen; i++)
	{
		int channel = i % 3;
		bool isLastTriplet = i >= dataLen - 3;

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
	case PackingType_VolByte:
	case PackingType_ChannelVolByte:
		tripletCount = std::min(maxTripletCount, (maxBytes - 2) / 3);
		break;
	case PackingType_PackedVol:
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
	case PackingType_VolByte:
		std::copy(pSource, pSource + tripletCount * 3, pDest);
		pDest += tripletCount * 3;
		break;
	case PackingType_ChannelVolByte:
		for (int i = 0; i < tripletCount; ++i)
		{
			*pDest++ = (uint8_t)(0 << 6) | pSource[3 * i + 0];
			*pDest++ = (uint8_t)(1 << 6) | pSource[3 * i + 1];
			*pDest++ = (uint8_t)(2 << 6) | pSource[3 * i + 2];
		}
		break;
	case PackingType_PackedVol:
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

uint8_t* chVolPack(PackingType packingType, uint8_t* pSource, int sourceLength, int romSplit, int& destLength)
{
	uint8_t* result = new uint8_t[2 * sourceLength + 500];
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
			auto pDestBefore = pDest;
			int tripletsConsumed = chVolPackChunk(pDest, pSource, tripletCount, romSplit, packingType);
			tripletCount -= tripletsConsumed;
			pSource += tripletsConsumed * 3;
			if (tripletCount > 0)
			{
				// Add padding if needed, but not on the last chunk
				int bytesEmitted = (int)(pDest - pDestBefore);
				int padding = romSplit - bytesEmitted;
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

//////////////////////////////////////////////////////////////////////////////
// RLE encodes a buffer. The method can do both a
// consecutive buffer or a buffer split in multiple 8kB buffers
//
uint8_t* rlePack(uint8_t* binBuffer, uint32_t length, int romSplit, int rleIncrement, int& resultLen)
{
	if (romSplit == 0)
	{
		printf("RLE encoding with no split\n");
		auto result = rleEncode(binBuffer, length, rleIncrement, resultLen);
		printf(
			"- Encoded %d volume commands (%d bytes) to %d bytes of data,\n"
			"  effective compression ratio %.2f%%\n",
			length,
			length / 2,
			resultLen,
			(1.0 * length / 2 - resultLen) / (length / 2) * 100);
		return result;
	}

	printf("RLE encoding with splits at %dKB boundaries", romSplit / 1024);

	uint8_t* destBuffer = new uint8_t[2 * length];
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
			auto bankSrc = binBuffer + 3 * tripletsEncoded;

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
			int lastPadding = romSplit - encodedLength;
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
		(1.0 * length / 2 - resultLen) / (length / 2) * 100);
	return destBuffer;
}


//////////////////////////////////////////////////////////////////////////////
// Converts a wav file to PSG binary format. The method can do both a
// consecutive buffer or a buffer split in multiple 8kB buffers
//
void convertWav(const std::string& filename, bool saveInternal, int costFunction, InterpolationType interpolation,
	int cpuFrequency, int dt1, int dt2, int dt3,
	int ratio, double amplitude, int romSplit, PackingType packingType, Chip chip, DataPrecision precision)
{
	// Load samples from wav file
	if (ratio < 1)
	{
		throw std::invalid_argument("Invalid number of inputs per output");
	}
	int frequency = cpuFrequency * ratio / (dt1 + dt2 + dt3);
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
	case Chip_AY38910:
		// MSX
		vol[0] = 0;
		for (int i = 1; i < 16; i++)
		{
			vol[i] = pow(2.0, i / 2.0) / pow(2.0, 7.5);
		}
		break;
	case Chip_SN76489:
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
	case DataPrecision_Float:
		binBuffer = viterbi<float>(ratio, amplitude, samples, samplesLen, dt1, dt2, dt3, interpolation, costFunction, saveInternal, binSize, vol);
		break;
	case DataPrecision_Double:
		binBuffer = viterbi<double>(ratio, amplitude, samples, samplesLen, dt1, dt2, dt3, interpolation, costFunction, saveInternal, binSize, vol);
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
	case PackingType_RLE:
		destBuffer = rlePack(binBuffer, binSize, romSplit, 1, destLength);
		break;
	case PackingType_RLE3:
		destBuffer = rlePack(binBuffer, binSize, romSplit, 2, destLength);
		break;
	case PackingType_VolByte:
	case PackingType_ChannelVolByte:
	case PackingType_PackedVol:
		destBuffer = chVolPack(packingType, binBuffer, binSize, romSplit, destLength);
		break;
	default:
		throw std::invalid_argument("Invalid packing type");
	}
	delete[] binBuffer;

	// Save the encoded buffer
	saveEncodedBuffer(filename + ".pcmenc", destBuffer, destLength);
	delete[] destBuffer;
}

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
		std::map<std::string, std::string>::const_iterator it = _args.find(name);
		if (it == _args.end())
		{
			return defaultValue;
		}
		return it->second;
	}

	int getInt(const std::string& name, uint32_t defaultValue)
	{
		std::map<std::string, std::string>::const_iterator it = _args.find(name);
		if (it == _args.end())
		{
			return defaultValue;
		}
		return atoi(it->second.c_str());
	}

	bool exists(const std::string& name) const
	{
		return _args.find(name) != _args.end();
	}
};


//////////////////////////////////////////////////////////////////////////////
// Program main.
//
int main(int argc, char** argv)
{
	try
	{
		Args args(argc, argv);

		std::string filename = args.getString("filename", "");
		int romSplit = args.getInt("r", 0) * 1024;
		bool saveInternal = args.exists("si");
		PackingType packingType = (PackingType)args.getInt("p", PackingType_RLE);
		int ratio = args.getInt("rto", 1);
		InterpolationType interpolation = (InterpolationType)args.getInt("i", Interpolation_Lagrange11);
		int costFunction = args.getInt("c", 2);
		int cpuFrequency = args.getInt("cpuf", 3579545);
		int amplitude = args.getInt("a", 115);
		int dt1 = args.getInt("dt1", 0);
		int dt2 = args.getInt("dt2", 0);
		int dt3 = args.getInt("dt3", 0);
		Chip chip = (Chip)args.getInt("chip", Chip_SN76489);
		DataPrecision precision = (DataPrecision)args.getInt("precision", DataPrecision_Float);

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
