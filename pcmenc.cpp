/*****************************************************************************
**
** Copyright (C) 2006 Arturo Ragozini, Daniel Vik
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

#include "st.h"

// Minimum allowed frequency difference for not doing frequency conversion
#define MIN_ALLOWED_FREQ_DIFF 0.005

#define MIN(a,b) fmin(a, b) // Beats std::max, same as ?:
#define MAX(a,b) fmax(a, b) // Beats std::max, same as ?:
#define ABS(a)   std::abs(a) // Beats fabs, ?:

#define str2ul(s) ((uint32_t)s[0]<<0|(uint32_t)s[1]<<8|(uint32_t)s[2]<<16|(uint32_t)s[3]<<24)
#define needSwap() (*(uint32_t*)"A   " == 0x41202020)

/* Lagrange's classical polynomial interpolation */
static double S(const double y[], int i, double dt,int L,int R)
{
    double  yd = 0;
    double  t = (double) i + dt;    
    
    for (int j=i+L; j<=i+R; j++)
    {
            double p = y[(j<0) ? 0 : j];
            for (int k=i+L; k<=i+R; k++)
            {
                if  (k!=j)
                    p*=(t-k)/(j-k);
            }
            yd+=p;
    }    
  return yd;
}

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
double* resample(double* in , uint32_t inLen,  uint32_t inRate, 
                 uint32_t outRate, uint32_t* outLen)
{
    uint32_t outBufLen = (uint32_t)((double)inLen * outRate / inRate) + 500;

    st_effect effp[1];

    bool swap = needSwap();

    st_sample_t* ibuf = (st_sample_t*)calloc(sizeof(st_sample_t), inLen);
    st_sample_t* obuf = (st_sample_t*)calloc(sizeof(st_sample_t), outBufLen);

    st_signalinfo_t iinfo = { inRate, 4, 0, 1, swap };
    st_signalinfo_t oinfo = { outRate, 4, 0, 1, swap };

    st_effect_t st_effect[1] = {
        {"resample", ST_EFF_RATE,
         st_resample_getopts, st_resample_start, st_resample_flow,
         st_resample_drain, st_resample_stop}
    };

    effp->h = st_effect;

    st_updateeffect(effp, &iinfo, &oinfo, 0);

    for (uint32_t i = 0; i < inLen; i++) {
        ibuf[i] = ST_FLOAT_DDWORD_TO_SAMPLE(in[i]);
    }
    char* argv[] = {"-ql"};
    st_resample_getopts(effp, 1, argv);
    st_resample_start(effp);

    st_size_t iLen = 0;
    st_size_t oLen = 0;

    for(;;) {
        st_size_t idone = ST_BUFSIZ;
        st_size_t odone = ST_BUFSIZ;
        int rv = st_resample_flow(effp, ibuf + iLen, obuf + oLen, &idone, &odone);
        iLen += idone;
        oLen += odone; 
        if (rv == ST_EOF || iLen + idone > inLen) {
            break;
        }
    }

    st_size_t odone = ST_BUFSIZ;
    st_resample_drain(effp, obuf + oLen, &odone);
    oLen += odone; 

    st_resample_stop(effp);

    double* outBuf = NULL;
    if (oLen > 0) {
        outBuf = (double*)calloc(sizeof(double), oLen);
        for (uint32_t i = 0; i < oLen; i++) {
            outBuf[i] = ST_SAMPLE_TO_FLOAT_DDWORD(obuf[i]);
        }
        *outLen = (uint32_t)oLen;
    }

    free(ibuf);
    free(obuf);
    
    return outBuf;
}


//////////////////////////////////////////////////////////////////////////////
// Loads a wav file and creates a new buffer with sample data.
//
double* loadSamples(const std::string& filename, uint32_t wantedFrequency, uint32_t* count)
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

    double* tempSamples = (double*)calloc(sizeof(double), sampleNum);

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
        retSamples = (double*)calloc(sizeof(double), sampleNum);
        memcpy(retSamples, tempSamples, sampleNum * sizeof(double));
        *count = sampleNum;
    }
    else 
	{
        printf("Resampling input wave from %dHz to %dHz\n", (int)samplesPerSec, (int)wantedFrequency);
        retSamples = resample(tempSamples, sampleNum, samplesPerSec, wantedFrequency, count);
    }

    free(tempSamples);

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

//////////////////////////////////////////////////////////////////////////////
// Encodes sample data to be played on the PSG.
// The output buffer needs to be three times the size of the input buffer
//
uint8_t* viterbi(int samplesPerTriplet, double amplitude, const double* samples, int length, 
               uint32_t idt1, uint32_t idt2, uint32_t idt3, 
               int interpolation, int costFunction,
               int saveInternal, uint32_t* binSize)
{
    double* y = (double*)calloc(sizeof(double), length + 256);

    double vol[16];
#ifdef MSX
	// MSX
	vol[0] = 0;
	for (int i = 1; i < 16; i++)
	{
		vol[i] = pow(2.0, i / 2.0) / pow(2.0, 7.5);
	}
#else
	// SMS
	for (int i = 0; i < 15; i++)
	{
		vol[i] = pow(10.0, -0.1*i);
	}
	vol[15] = 0.0;
#endif

    double inputMin = 1.0E50;
    double inputMax = -1.0E50;

    for (int i = 0; i < length; i++)
	{
        inputMin = MIN(samples[i], inputMin);
        inputMax = MAX(samples[i], inputMax);
    }

    for (int i = 0; i < length; i++)
	{
        y[i] = amplitude * (samples[i] - inputMin) / (inputMax - inputMin);
    }
    for (int i = length; i < length + 256; i++)
	{
        y[i] = y[length - 1];
    }

    double dt[3];
	uint32_t cyclesPerTriplet = idt1 + idt2 + idt3;
    dt[0] = (double)idt1 / cyclesPerTriplet;
    dt[1] = (double)idt2 / cyclesPerTriplet;
    dt[2] = (double)idt3 / cyclesPerTriplet;

    if (samplesPerTriplet < 1)
	{
        samplesPerTriplet = 1;
    }

    printf("Viterbi SNR optimization:\n");
    printf("   %d input samples per PSG triplet output\n", samplesPerTriplet);
    printf("   dt1 = %d  (Normalized: %1.3f)\n", (int)idt1, dt[0]);
    printf("   dt2 = %d  (Normalized: %1.3f)\n", (int)idt2, dt[1]);
    printf("   dt3 = %d  (Normalized: %1.3f)\n", (int)idt3, dt[2]);

    int     N = (length + samplesPerTriplet - 1) / samplesPerTriplet * 3;
    double* x = (double*)calloc(sizeof(double), N);

    int     nL,nR;

    switch (interpolation) {
    case 0:
        {
            printf("   Resampling using Linear interpolation\n");
            nL=0;
            nR=1;
        }
        break;
    case 1:
        {
            printf("   Resampling using Quadratic interpolation\n");
            nL=0;
            nR=2;
        }
        break;
    case 2:
    default:
        {
            printf("   Resampling using Lagrange interpolation on 11 points\n");
            nL=-5;
            nR=5;
        }
        break;
    }
    for (int i = 0; i < N / 3; i++) {
        int     t0 = (samplesPerTriplet * i);
        double  t1 = (samplesPerTriplet * (i+dt[0]));
        double  t2 = (samplesPerTriplet * (i+dt[0]+dt[1]));
        double  dt1 = t1-(int)t1;
        double  dt2 = t2-(int)t2;

        x[3*i+0] =  y[t0];
        x[3*i+1] =  S(y,(int)t1,dt1,nL,nR);
        x[3*i+2] =  S(y,(int)t2,dt2,nL,nR);
    }

    if (saveInternal) 
	{
		dump("x.bin", (uint8_t*)x, N * sizeof(double));
		dump("y.bin", (uint8_t*)y, (length * 256) * sizeof(double));
    }

	uint8_t nxtS[16*16*16];
	for (int i = 0; i < 16; i++) 
	{
		for (int j = 0; j < 16; j++) 
		{
			for (int in = 0; in < 16; in++) 
			{
				nxtS[i << 8 | j << 4 | in] = (uint8_t)(j * 16 + in);
			}
		}
	}

	double curV[16*16*16];
	for (int i = 0; i < 16; i++) 
	{
		for (int j = 0; j < 16; j++) 
		{
			for (int in = 0; in < 16; in++) 
			{
				curV[i << 8 | j << 4 | in] = vol[i] + vol[j] + vol[in];
			}
		}
	}

    uint8_t* Stt[256];
    uint8_t* Itt[256];
    double L[256];
    uint8_t  St[256];
    uint8_t  It[256];

    for (int i = 0; i < 256; i++)
	{
        Stt[i] = (uint8_t*)calloc(sizeof(uint8_t), N);
        Itt[i] = (uint8_t*)calloc(sizeof(uint8_t), N);
        L[i] = 0;
        St[i] = 1;
        It[i] = 1;
    }

    printf("   Using cost function: L%d\n",costFunction);

    for (int t = 0; t < N; t++) 
	{
        double Ln[256];
        for (int i = 0; i < 256; i++) 
		{
            Ln[i] = 1.0E50; //inf
        }

        if (t % 1024 == 0) 
		{
            printf("Processing %3.2f%%\r", 100. * t / N);
        }

		for (int i = 0; i < 16 * 16 * 16; ++i)
		{
			int cs = i >> 4;
			int in = i & 15;

			double cv = curV[i];
			int ns = nxtS[i];

			double Ltst;
            double normVal = ABS(x[t] - cv);
            switch (costFunction) 
			{
            case 1:
                Ltst = L[cs] + dt[t % 3] * normVal;
                break;
            case 2:
                Ltst = L[cs] + dt[t % 3] * normVal * normVal;
                break;
            case 3:
                Ltst = L[cs] + dt[t % 3] * normVal * normVal * normVal;
                break;
            case 4:
                Ltst = L[cs] + dt[t % 3] * normVal * normVal * normVal * normVal;
                break;
            default:
                Ltst = L[cs] + dt[t % 3] * pow(normVal, costFunction);
                break;
            }
             
            if (Ln[ns] >= Ltst)
			{
                Ln[ns] = Ltst;
                St[ns] = (uint8_t)cs;
                It[ns] = (uint8_t)in;
            }
        }

        for (int i = 0; i < 256; i++)
		{
            L[i]      = Ln[i];
            Stt[i][t] = St[i];
            Itt[i][t] = It[i];
        }
    }
	
	printf("Processing %3.2f%%\n", 100.0);

	int minIndex = 0;
    for (int i = 0; i < 256; i++)
	{
        if (L[minIndex] > L[i])
		{
            minIndex = i;
        }
    }

    printf("\nThe cost metric in Viterbi is about %3.3f\n\n", L[minIndex]);

    uint8_t* P = (uint8_t*)calloc(sizeof(uint8_t), N);
    uint8_t* I = (uint8_t*)calloc(sizeof(uint8_t), N);

    P[N - 1] = Stt[minIndex][N - 1];
    I[N - 1] = Itt[minIndex][N - 1];
    for (int t = N - 2; t >= 0; t--)
	{
        P[t] = Stt[P[t + 1]][t];
        I[t] = Itt[P[t + 1]][t];
    }


    double* V = (double*)calloc(sizeof(double), N);
    
    for (int t = 0; t <N; t++)
	{
		V[t] = curV[P[t] << 4 | I[t]];
    }

    if (saveInternal)
	{
		dump("v.bin", (uint8_t*)V, N * sizeof(double));
    }    


/* Compute the SNR independently from the cost metric used in Viterbi */

    double en = 0;
    double er = 0;
    double mi = 0;    
    for (int i = 0; i < N / 3; i++)
	{
        en += (x[3 * i + 0]) * (x[3 * i + 0]) * dt[0] +
              (x[3 * i + 1]) * (x[3 * i + 1]) * dt[1] +
              (x[3 * i + 2]) * (x[3 * i + 2]) * dt[2];
        er += (x[3 * i + 0]-V[3 * i + 0]) * (x[3 * i + 0]-V[3 * i + 0]) * dt[0] +
              (x[3 * i + 1]-V[3 * i + 1]) * (x[3 * i + 1]-V[3 * i + 1]) * dt[1] +
              (x[3 * i + 2]-V[3 * i + 2]) * (x[3 * i + 2]-V[3 * i + 2]) * dt[2];
        mi += (x[3 * i + 0]) * dt[0] + (x[3 * i + 1]) * dt[1] + (x[3 * i + 2]) * dt[2];
    }
    
    double  var = en - mi*mi*3/N;
    printf("SNR is about %3.2f\n", 10 * log10( var / er ));

    free(y);
    free(x);
    free(P);
    free(V);
       
    for (int i = 0; i < 256; i++)
	{
        free(Stt[i]);
        free(Itt[i]);
    }

    *binSize = N;
    return I;
}


//////////////////////////////////////////////////////////////////////////////
// RLE encodes a PSG sample buffer. The encoded buffer is created and returned
// by the function.
//
uint8_t* rleEncode(const uint8_t* buffer, int length, int incr, uint32_t* encLenth)
{
    const uint8_t* I = buffer;

    uint8_t* sRet = (uint8_t*)calloc(sizeof(uint8_t), 2 * length);
    sRet[0] = ((length / 3) >> 0) & 0xff;
    sRet[1] = ((length / 3) >> 8) & 0xff;

    uint8_t* s = sRet + 2;

    int j = 3;
    int A[3] = {I[0], I[1], I[2]};
    int la[3] = {0, 0, 0};
    int ja[3] = {0, 1, 2};

    for (int i = 3; i < length; i++) {
        int x = i % 3;

        if (A[x] == I[i] && la[x] < 15 - (incr - 1) && i < length - 3) {
            la[x] += incr;
        }
        else {
            s[ja[x]] = (uint8_t)(la[x] << 4 | A[x]);
            la[x] = 0;
            ja[x] = j++;
            A[x] = I[i];
            if (i >= length - 3) 
			{
                s[ja[x]] = (uint8_t)(la[x] << 4 | A[x]);
            }
        }
    }

    *encLenth = j + 2;

    return sRet;
}

//////////////////////////////////////////////////////////////////////////////
// Saves an encoded buffer, the file extension is replaced with .bin.
//
void saveEncodedBuffer(const std::string& filename, const uint8_t* buffer, int length)
{
	dump(filename, buffer, length);
}

uint8_t* chVolPack(int type, uint8_t* binBuffer, uint32_t length, int romSplit, uint32_t* destLength)
{
    uint8_t* destBuffer = (uint8_t*)calloc(sizeof(uint8_t), 2 * length + 500);
    uint8_t* destP = destBuffer;
    if (!romSplit) {
        *destP++ = (uint8_t)((length >> 0) & 0xff);
        *destP++ = (uint8_t)((length >> 8) & 0xff);
        for (uint32_t i = 0; i < length; i++) {
            if (type == 0) {
                *destP++ = binBuffer[i];
            }
            else {
                *destP++ = (uint8_t)((i % 3) << 6) | binBuffer[i];
            }
        }
    }
    else {
        int channel = 0;
        do 
		{
            uint32_t subLength = std::min(length, (uint32_t)0x2000 * 2 - 2);
            *destP++ = (uint8_t)((subLength >> 0) & 0xff);
            *destP++ = (uint8_t)((subLength >> 8) & 0xff);
            for (uint32_t i = 0; i < subLength; i++) 
			{
                if (type == 0) 
				{
                    *destP++ = (uint8_t)(channel << 6) | *binBuffer++;
                    channel = (channel + 1) % 3;
                }
                else {
                    *destP++ = *binBuffer++;
                }
            }
            length -= subLength;
        } while (length > 0);
        
        while ((destP - destBuffer) & 0x1fff) {
            *destP++ = 0;
        }
    }
    *destLength = (uint32_t)(destP - destBuffer);
    return destBuffer;
}

//////////////////////////////////////////////////////////////////////////////
// RLE encodes a buffer. The method can do both a
// consecutive buffer or a buffer split in multiple 8kB buffers
//
uint8_t* rlePack(uint8_t* binBuffer, uint32_t length, int romSplit, int incr, uint32_t* destLength)
{
    if (!romSplit) {
        printf("Encoding samples for original player\n");
        return rleEncode(binBuffer, length, incr, destLength);
    }

    printf("Encoding samples for rom player\n");
    
    uint8_t* destBuffer = (uint8_t*)calloc(sizeof(uint8_t), 2 * length);
    int srcOffset = 0;
    const uint32_t SUB_SAMPLE_LEN = 10000 * 2;
    *destLength = 0;

    // For rom version of replayer
    uint32_t count = length / 3;
    while (count > 0) {
        int curCount = std::min(count, SUB_SAMPLE_LEN);

        uint8_t* encBuffer;
        uint32_t encLen = 0;

        for (;;) {
            encBuffer = rleEncode(binBuffer + 3 * srcOffset, curCount * 3, incr, &encLen);       
            if (encLen <= 0x2000 * 2) {
                break;
            }
            free(encBuffer);
            curCount = 995 * curCount / 1000;
        }
        memcpy(destBuffer + *destLength, encBuffer, encLen);
        memset(destBuffer + *destLength + encLen, 0, 0x2000 * 2 - encLen);

        free(encBuffer);

        count -= curCount;
        *destLength += 0x2000 * 2;
        srcOffset += curCount;
    }

    return destBuffer;
}


//////////////////////////////////////////////////////////////////////////////
// Converts a wav file to PSG binary format. The method can do both a
// consecutive buffer or a buffer split in multiple 8kB buffers
//
int convertWav(const std::string& filename, int saveInternal, int costFunction, int interpolation,
               uint32_t cpuFrequency, uint32_t dt1, uint32_t dt2, uint32_t dt3, 
               int encodingType, int ratio, double amplitude, int romSplit, int packingType)
{
    if (encodingType != 0) {
        printf("Encoding type %d not supported\n", encodingType);
    }
    // Load samples from wav file
    if (ratio < 1) {
        printf("Invalid number of inputs per output\n");
        return 0;
    }
    uint32_t count;
    uint32_t frequency = cpuFrequency * ratio / (dt1 + dt2 + dt3);
    if (frequency == 0) {
        printf("Invalid frequency\n");
        return 0;
    }

    printf("Encoding PSG samples at %dHz\n", (int)frequency);

    double* samples = loadSamples(filename, frequency, &count);
    if (samples == NULL) {
        printf("Failed to load wav file: %s\n", filename.c_str());
        return 0;
    }
    
    // Do viterbi encoding
    uint32_t binSize = 0;
    uint8_t* binBuffer = viterbi(ratio, amplitude, samples, count, dt1, dt2, dt3, 
                               interpolation, costFunction, saveInternal, &binSize);

    // RLE encode the buffer. Either as one consecutive RLE encoded
    // buffer, or as 8kB small buffers, each RLE encoded with header.
    uint8_t* destBuffer = 0;
    uint32_t destLength = 0;

    switch (packingType) {
    case 0: //4 bit RLE
        destBuffer = rlePack(binBuffer, binSize, romSplit, 1, &destLength);
        break;
    case 1: //3 bit RLE
        destBuffer = rlePack(binBuffer, binSize, romSplit, 2, &destLength);
        break;
    case 2:
        destBuffer = chVolPack(0, binBuffer, binSize, romSplit, &destLength);
        break;
    case 3:
        destBuffer = chVolPack(1, binBuffer, binSize, romSplit, &destLength);
        break;
    }

    // Save the encoded buffer
    saveEncodedBuffer(filename + ".pcmenc", destBuffer, destLength);
    free(destBuffer);
    free(binBuffer);

    return 1;
}


//////////////////////////////////////////////////////////////////////////////
// Program main.
//
int main(int argc, char** argv)
{
	std::string filename;
    int romSplit = 0;
    int saveInternal = 0;    
    int packingType = 0;  // 0=RLE, 1=TEST
    int encodingType = 0;
    int ratio = 1;
    int interpolation = 2;
    int costFunction = 2;
    uint32_t cpuFrequency = 3579545;
    uint32_t amplitude = 115;
    uint32_t dt1 = (uint32_t)-1;
    uint32_t dt2 = (uint32_t)-1;
    uint32_t dt3 = (uint32_t)-1;

    // Parse command line options
    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], "-cpuf") || 0 == strcmp(argv[i], "/cpuf")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            cpuFrequency = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-a") || 0 == strcmp(argv[i], "/a")) {
            if (++i >= argc) {
				filename.clear();
				break;
            }
            amplitude = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-dt1") || 0 == strcmp(argv[i], "/dt1")) {
            if (++i >= argc) {
				filename.clear();
				break;
            }
            dt1 = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-dt2") || 0 == strcmp(argv[i], "/dt2")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            dt2 = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-dt3") || 0 == strcmp(argv[i], "/dt3")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            dt3 = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-e") || 0 == strcmp(argv[i], "/e")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            encodingType = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-rto") || 0 == strcmp(argv[i], "/rto")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            ratio = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-r") || 0 == strcmp(argv[i], "/r")) {
            romSplit = 1;
        }
        else if (0 == strcmp(argv[i], "-si") || 0 == strcmp(argv[i], "/si")) {
            saveInternal = 1;
        }
        else if (0 == strcmp(argv[i], "-p") || 0 == strcmp(argv[i], "/p")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            packingType = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-c") || 0 == strcmp(argv[i], "/c")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            costFunction = atoi(argv[i]);
        }
        else if (0 == strcmp(argv[i], "-i") || 0 == strcmp(argv[i], "/i")) {
            if (++i >= argc) {
                filename.clear();
                break;
            }
            interpolation = atoi(argv[i]);
        }
        else if (filename.empty() && argv[i][0] != '-') {
            filename = argv[i];
        }
        else {
            filename.clear();
            break;
        }
    }

    // Set defaults for dt depending on encoding type
    switch (ratio) {
    default:
    case 0:
        dt1 = dt1 != (uint32_t)-1 ? dt1 : 32;
        dt2 = dt2 != (uint32_t)-1 ? dt2 : 27;
        dt3 = dt3 != (uint32_t)-1 ? dt3 : 266;
        break;
    case 1:
        dt1 = dt1 != (uint32_t)-1 ? dt1 : 156;
        dt2 = dt2 != (uint32_t)-1 ? dt2 : 27;
        dt3 = dt3 != (uint32_t)-1 ? dt3 : 141;
        break;
    case 2:
        dt1 = dt1 != (uint32_t)-1 ? dt1 : 73;
        dt2 = dt2 != (uint32_t)-1 ? dt2 : 84;
        dt3 = dt3 != (uint32_t)-1 ? dt3 : 87;
        break;
    }

    if (dt1 + dt2 + dt3 == 0 || cpuFrequency == 0) 
	{
        filename.clear();
    }

    if (filename.empty()) 
	{
        printf("Usage:\n");
        printf("pcmenc.exe [-r] [-e <encoding>] [-cpuf <freq>] [-p <packing>]\n");
        printf("           [-dt1 <tstates>] [-dt2 <tstates>] [-dt3 <tstates>]\n");
        printf("           [-a <amplitude>] [-rto <ratio>] <wavfile>\n");
        printf("\n");
        printf("    -r              Pack encoded wave into 8kB blocks for rom replayers\n");
        printf("\n");
        printf("    -p <packing>    Packing type:                b7...b5|b4...b0\n");
        printf("                        0 = 4bit RLE (default)   run len|PSG vol\n");
        printf("                        1 = 3 bit RLE; as before but b5 =0\n");
        printf("                        1 = 1 byte vol\n");
        printf("                        2 = 1 byte {ch, vol} pairs\n");
        printf("\n");
        printf("    -cpuf <freq>    CPU frequency of the CPU (Hz)\n");
        printf("                        Default: 3579545\n");
        printf("\n");
        printf("    -dt1 <tstates>  CPU Cycles between update of channel A and B\n");
        printf("    -dt2 <tstates>  CPU Cycles between update of channel B and C\n");
        printf("    -dt3 <tstates>  CPU Cycles between update of channel C and A\n");
        printf("                    The replayer sampling base period is \n");
        printf("                          T = dt1+dt2+dt3\n");
        printf("                    Defaults (depends on rto):\n");
        printf("                        ratio = 1 : dt1=32, dt2=27,  dt3=266 => 11014Hz\n");
        printf("                        ratio = 2 : dt1=156,dt2=27,  dt3=141 => 22096Hz \n");
        printf("                        ratio = 3 : dt1=73, dt2=84,  dt3=87  => 44011Hz\n");
        printf("\n");
        printf("                    Note that the replayed sampling base period depends\n");
        printf("                    on the replayer and how many samples it will play\n");
        printf("                    in each PSG tripplet update. The default settings\n");
        printf("                    are based on:\n");
        printf("                        1 : replayer_core11025 which plays one sample per\n");
        printf("                            psg tripplet update\n");
        printf("                        2 : replayer_core22050 which plays two sample per\n");
        printf("                            psg tripplet update\n");
        printf("                        3 : replayer_core44100 which plays three sample\n");
        printf("                            per psg tripplet update \n");
        printf("\n");
        printf("    -a <amplitude>  Input amplitude before encoding.\n");
        printf("                        Default 115\n");
        printf("\n");
        printf("    -rto <ratio>   Number of input samples per PSG triplet\n");
        printf("                        Default: 1\n");
        printf("\n");
        printf("                   This parameter can be used to oversample the input\n");
        printf("                   wave. Note that this parameter also will affect the\n");
        printf("                   replay rate based on how many samples per PSG tripplet\n");
        printf("                   update the replayer uses.\n");
        printf("\n");
        printf("    -c <costfun>    Viterbi cost function:\n");
        printf("                        1   : ABS measure\n");
        printf("                        2   : Standard MSE (default)\n");
        printf("                        > 2 : Lagrange interpolation of order 'c'\n");
        printf("\n");
        printf("    -i <interpol>   Resampling interpolation mode:\n");
        printf("                        0 = Linear interpolation\n");
        printf("                        1 = Quadratic interpolation\n");
        printf("                        2 = Lagrange interpolation (default)\n");
        printf("\n");
        printf("    <wavfile>       Filename of .wav file to encode\n");
        printf("\n");

        return 0;
    }

    return convertWav(filename, saveInternal, costFunction, interpolation, cpuFrequency, dt1, dt2, dt3, encodingType, ratio, (double)amplitude / 100, romSplit, packingType);
}


// */