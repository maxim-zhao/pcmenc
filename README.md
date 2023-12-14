pcmenc
======

[![Build status](https://ci.appveyor.com/api/projects/status/777xka4fol7lbxa7?svg=true)](https://ci.appveyor.com/project/maxim-zhao/pcmenc)

This is a clone of [pcmenc](http://www.dvik-joyrex.com/tools.html), a program which generates high-quality sample data plus matching assembly code players for MSX. 

The features I have added are:

* Support for targetting the SN76489 audio chip, as found in:
  * Sega Master System
  * Sega Game Gear
  * Sega Game 1000
  * Sega Computer 3000
  * Sega Megadrive/Genesis/Mega Drive
  * ColecoVision
  * BBC Micro
  * Lots of arcade games and esoteric home computers
* Optimal bank packing
* Player code for regular Z80 chips, targeting popular sampling rates and CPU clocks
* Some speedups, possibly MSVC specific
* A 64-bit build to allow processing longer files (the process consumes huge amounts of RAM)
* Support for packed 4-bit data, which often wins over RLE anyway
* Support for vector packing (using https://github.com/genbattle/dkm) - which tends to produce much lower quality output, so it is not recommended
* Support for skewing the sample to improve the potential resolution, as the outputs are non-linear. This is based on work by blargg in [wav_to_psg](https://github.com/maxim-zhao/wav_to_psg).

Usage
-----

First, decide on your playback rate and data rate requirements. This will usually be in the form of a playback routine. The playback routines are mapped to specific encoder settings:

1. Sampling rate. Just like on any other sample player, more samples per second = better reproduction, especially of higher frequencies. This maps to the interaction between the `-dt` (inter-output timings) and `-clock` (CPU speed) settings.
2. `-rto`: The ratio of input samples per full update (three channels) of the PSG. This is from 1 (highest accuracy) to 3 (lowest). This maps somewhat to the concept of bit depth for audio: higher accuracy means clearer audio. However, there's a linear relationship to the data size: `-rto 3` will emit 1/3 as much data as `-rto 1`. Also, there is not enough CPU time (on the Master System at least) to emit three values to the PSG for every sample at ~44kHz.
3. `-p`: packing. There are many packing routines but in general, the "packed volumes" `-p4` option is the simplest and fastest: it packs each PSG update into half of a byte. The RLE packing `-p0` often results in larger data. 

In general, trading off lower accuracy (`-rto`) for a higher sampling rate gets you better quality; `-rto 3` is generally the best option if you have high quality audio.

Next, prepare your audio file to match this playback rate. Perform all the manipulations it needs (amplifying, compressing, sample rate conversion) in a half-decent audio editor. I use [Audacity](http://www.audacityteam.org/), which is good enough.

Then, ideally, find the commandline for conversion from the comments in the player code and convert using that. It'll be something like this:

```
pcmenc -dt1 81 -dt2 81 -dt3 81 -rto 3 -r 16 input.wav
```

A file will be produced with the suffix `.pcmenc`, e.g. `input.wav.pcmenc`. Binary include it in your program and invoke your player code according to its comments, and marvel at the quality of the audio!

Use the `-r` parameter to split data into chunks. You will need to play these chunks in order by calling the playback function repeatedly.

Use the `-smooth` parameter to apply a low-frequency offset to the data to improve sample resolution. The logarithmic outputs of the PSG means that it has poor resolution at the maximum value of a wave and good resolution near the minimum. This offset means that non-maximum-amplitude parts of the audio will be skewed towards the area with best resolution. You may need to experiment to find a value that suits the sample rate you are using.

You may notice that the replayers do not map to the playback rates you might normally see like 8000Hz, 44100Hz, 48000Hz, etc. This is because the playback routines end up using a number of CPU cycles per input sample which cannot be a perfect match. The closest we can get to 44100Hz output on a 3579545Hz CPU is to emit a sample every 81 cycles (2/3 of the time) or 82 cycles (every third sample), which corresponds to an average rate of 44010.8 outputs per second. The difference is generally imperceptible.

Commandline options
-------------------

| Flag | Description |
|------|-------------|
|`-r <n>` |Pack encoded wave into `n`KB blocks for rom replayers (typically 16) |
|`-p <packing>` |Packing type:<ul><li>0 = 4bit RLE (default): bits 7-4 = run length, bits 3-0 = value<li>1 = 3 bit RLE; as before but run length is in bits 7-5<li>2 = 1 byte values, upper nibble unused<li>3 = 1 byte {channel, value} pairs<li>4 = big-endian packed {value, value} pairs<li>5 = K-means clustered vector tables (6 values per vector)<li>6 = K-means clustered vector tables (3 values per vector)</ul> |
|`-cpuf <frequency>` |CPU frequency (Hz), default 3579545 |
|`-dt1 <cycles>` |CPU cycles between update of channel A and B |
|`-dt2 <cycles>` |CPU cycles between update of channel B and C |
|`-dt3 <cycles>` |CPU cycles between update of channel C and A |
|`-smooth <amount>` |Low-frequency skewing adjustment decay rate. Default 0 = off. 10 is suitable for 44kHz audio |
|`-a <amplitude>` |Overdrive amplitude adjustment. Default 100 = no adjustment. |
|`-rto <ratio>` |Number of input samples per PSG triplet. Default 1.
|`-c <costfun>` |Viterbi cost function: <ul><li>1 = abs() measure<li>2 = Standard mean square estimate (default)<li>&gt;2 = Lagrange interpolation of order 'c' |
|`-i <interpol>` |Resampling interpolation mode: <ul><li>0 = Linear interpolation<li>1 = Quadratic interpolation<li>2 = Lagrange interpolation (default) |
|`-chip <chip>` |Chip type:<ul><li>0 = AY-3-8910/YM2149F (MSX sound chip)<li>1 = SN76489/SN76496/NCR8496 (SMS sound chip) (default) |

Example
-------

```
>pcmenc.exe -rto 3 -p 4 -dt1 81 -dt2 81 -d
t3 82 -r 16 -smooth 10 Knight Rider.wav"
Encoding PSG samples at 44010Hz
Loading Knight Rider.wav...done; 3372462 samples
Skewing samples for better quality...done
Viterbi SNR optimization:
   3 input samples per PSG triplet output
   dt1 = 81  (Normalized: 0.332)
   dt2 = 81  (Normalized: 0.332)
   dt3 = 82  (Normalized: 0.336)
   Using 4 bytes data precision
   Resampling using Lagrange interpolation on 11 points... done (3372462 output points)
   Using cost function: L2
Processing 100.00%
The cost metric in Viterbi is about 80.164
SNR is about 25.38
Converted 3372462 samples to 3372462 outputs in 45.05s = 74869 samples per second
Packing data with splits at 16KB boundaries, as packed volume pairs %aaaabbbb
Packed as 1686488 bytes of data (103 banks with 0 bytes padding)
Saving 1686488 bytes to Knight Rider.wav.pcmenc...done
```

How it works
------------

* First, the sample is loaded into memory and converted to mono. 
* It is then normalised to the range 0..1, by default in a way that clips the peaks to get a louder result. 
* Next, for each sample:
  * It iterates through all possible PSG volume states (including ones that are equivalent output but different channel values or ordering).
  * For each possible state of "the other two channels", it selects the value for "the third channel" which will minimise the total error between the final output and the desired output. This error is defined by a configurable cost metric (by default, the time-weighted sum of squares of the deviation).
  * This acts to reduce the search space from 16^(number of output values) - which it is not feasible to explore on a normal computer for data of much length - to merely 8192 × (number of output values). However, it also requires 512 × (number of output values) bytes of memory, which may mean more memory than your computer has if the audio files are long - beware!
* The original authors describe it as a viterbi algorithm but I'm not sure it is...
* Once this is done, it is able to select the final value with the minimum total error and back-track it to a series of values to produce this minimised error
* The resulting stream is then passed into various packing methods to produce something amenable to use in real life:
  * Packing into ROM banks
  * Compressing (sometimes not effectively)
  * Adding metadata to know when the end is reached

Original readme
---------------

Here's the original README, some of which no longer applies:

``` 
  =================================================================
  PCMENC - The Encoder for Crystal Clear PSG samples - Version 0.01
  =================================================================
  
               (c) 2006 Arturo Ragozini, Daniel Vik


Introduction
------------

This package contains an encoder application and replayers to create
and play crystal clear PSG samples on your MSX. 

Each input sample is approximated by the sum of 3 PSG channels. As the
z80 can vary  only one channel at time, the encoder, using a Viterbi
search, finds the best sequence of PSG channel variations that
approximate
(MSE) the input sequence. The output of pcmenc is a sequence of PSG
levels for the 3 channels that best approximate the input sequence. 

Provided that the ASM replayer can change the PSG levels at a fixed
rate, you can configure the encoder at least in 3 ways:

1) spending three PSG transitions per sample in input: this leads to the
best SNR (the lowest quantization noise) but limits the sampling
frequency to 11,025kHz. The unpacked encoding rate is 3x4=12bit per
sample. 

2) spending two PSG transitions per input sample: you have average SNR
and you can rise the sampling frequency up to 22,050kHz. The unpacked
encoding rate is 3x4/2=6bit per sample.

3) one PSG transition per input sample: minimum quality in exchange of
maximum sampling frequency. Using the provided replayer you can reach
44100kHz. The unpacked encoding rate is 4bits per sample.

Moreover the encoder (and the replayers), supports multiple types of
packing. By default, the he PSG levels are RLE packed and can be
optionally segmented in chunks in order to be used applications that use
multiple memory pages.

Although pcmenc.exe is able to do frequency conversions (upsampling and
downsampling), its suggested to use a 8bit (or 16bit) mono source wave
with the same frequency as the replayer frequency. In this case no
frequency conversion occurs. To get good results some preprocessing of
the wave file may be required.





Usage of pcmenc.exe
-------------------

Usage:
pcmenc.exe [-r] [-e <encoding>] [-cpuf <freq>] [-p <packing>]
           [-dt1 <tstates>] [-dt2 <tstates>] [-dt3 <tstates>]
           [-a <amplitude>] [-rto <ratio>] <wavfile>

    -r              Pack encoded wave into 8kB blocks for rom replayers

    -p <packing>    Packing type:                b7...b5|b4...b0
                        0 = 4bit RLE (default)   run len|PSG vol
                        1 = 3 bit RLE; as before but b5 =0
                        1 = 1 byte vol
                        2 = 1 byte {ch, vol} pairs

    -cpuf <freq>    CPU frequency of the CPU (Hz)
                        Default: 3579545

    -dt1 <tstates>  CPU Cycles between update of channel A and B
    -dt2 <tstates>  CPU Cycles between update of channel B and C
    -dt3 <tstates>  CPU Cycles between update of channel C and A
                    The replayer sampling base period is 
                          T = dt1+dt2+dt3
                    Defaults (depends on rto):
                        ratio = 1 : dt1=32, dt2=27,  dt3=266 => 11014Hz
                        ratio = 2 : dt1=156,dt2=27,  dt3=141 => 22096Hz 
                        ratio = 3 : dt1=73, dt2=84,  dt3=87  => 44011Hz
                        
                    Note that the replayed sampling base period depends
                    on the replayer and how many samples it will play
                    in each PSG tripplet update. The default settings
                    are based on:
                        1 : replayer_core11025 which plays one sample per
                            psg tripplet update
                        2 : replayer_core22050 which plays two sample per
                            psg tripplet update
                        3 : replayer_core44100 which plays three sample
                            per psg tripplet update 

    -a <amplitude>  Input amplitude before encoding.
                        Default 115

    -rto <ratio>   Number of input samples per PSG triplet
                        Default: 1
                   
                   This parameter can be used to oversample the input
                   wave. Note that this parameter also will affect the
                   replay rate based on how many samples per PSG tripplet
                   update the replayer uses.

    -c <costfun>    Viterbi cost function:
                        1   : ABS measure
                        2   : Standard MSE (default)
                        > 2 : Lagrange interpolation of order 'c'

    -i <interpol>   Resampling interpolation mode:
                        0 = Linear interpolation
                        1 = Quadratic interpolation
                        2 = Lagrange interpolation (default)
                                
    <wavfile>       Filename of .wav file to encode
    
    
The values of dt1, dt2, and dt3 defines the time (in T-states) between
the PSG channel updates in the replayer. The sum of the dt's values is
the base sample period of the replayer. You can fit one, two, or three
PSG transitions per inputs sample depending on the ratio between the
frequency input and the replayer base frequency. The frequency of the
replayer is:

                           CPU frequency
   replayed frequency  =  ---------------
                          dt1 + dt2 + dt3
        
The default values are chosen to match the replayer for the specific
encoding that comes with the pcmenc package and the frequency of the
replayed samples is by default either 11.025Hz, 22.05Hz or 44.1Hz but
this can be controlled by the dt arguments. 


Ratio 1 - One input sample per PSG output triplet
Each sample in input is approximated by 3 PSG transitions: this solution
gives the best SNR as the PSG can vary its 3 channels each sample in
input (human voice, low pass sounds and bass music it can reach 42dB of
SNR). On a standard MSX the sampling frequency cannot be much higher
than 8KHz,moreover each samples needs  12bit to be stored, do not expect
miracles from RLE.

Ratio 2 - Two input sample per PSG output triplet
Each couple of samples in input is approximated by 3 PSG transitions:
this
solution has lower SNR, as for odd samples the PSG can vary 2 channels
while on even samples the PSG vary the last channel (for "well
behavioured" inputs you can have 36dB of SNR). On a standard MSX the
sampling frequency cannot be much higher than 22kHz. Each samples needs
6bit to be stored, as usual, do not expect miracles from RLE.

Ratio 3 - Three input sample per PSG output tripplet
Each samples in input is approximated by 1 PSG transitions: this
solution has the lowest SNR (for "well behavioured" inputs you can have
26dB of SNR) but, on a standard MSX, the sampling frequency can rise up
to 44kHz.
Each samples needs 4bit to be stored and as usual, do not expect
miracles from RLE. Due to the special time constraints the replayer that
can support 44kHz requires 3bit RLE (instead of the 4bit RLE).


Ratio > 3 - Oversampled input
A ratio > 3 will result in an oversampled input. In these cases, the
ratio should be a multiple of the desired input sample per PSG output
tripplet. When using a ratio > 3, the dt1, dt2, and dt3 values needs
to be set in the argument list.

To get the best result its good to have an input wave file to the
encoder that matches the replayed frequency. The encoder does down and
upsampling of the input wave to match the replayed frequency, but the
current implementation does not do antialiasing which most likely is
needed to get a good result. So the recommended process is:

1. Find a wave file you'd like to have played on the MSX
2. Downsample it to the desired playback frequency, e.g. 11kHz with
antialiasing 
   (i.e. a 5.5kHz lowpass filter)
3. Run the encoder with the downsampled wave file as input.
4. Change the incbin statement in vplayer.asm to the filename of the
generated .bin file.
5. Assemble vplayer.asm with sjasm (or other compatible assembler)
6. Run the generated .com or .rom file in your MSX
    

Examples for creating .com readable samples with pcmenc.exe
-----------------------------------------------------------

The replayer cores show the preferred command line options to be used
by the replayer you wish to use.

To encode a file for the .com version of the replayer:

    PcmEnc.exe sample.wav

This outputs a file called sample.bin which can be compiled with
vplayer.asm:

    sjasm vplayer.asm

The ExeType in vplayer.asm should be set to 0 to create a .com file. 

The com file can then be loaded and played in your MSX. Note that this
replayer is only able to play short samples that fits in 64kB ram.


Examples for creating ascii8 .rom readable samples with pcmenc.exe
------------------------------------------------------------------

To encode a file for the .com version of the replayer:

PcmEnc.exe -r sample.wav

The -r option will split the sample data into 8kB blocks to make it
playable with the rom version of the replayer (not provided). The 
output is a file called sample.bin which can be compiled with 
vplayer.asm:

    sjasm vplayer.asm
    
The ExeType should be set to 0 to create an ascii8 .rom file. 

The com file can then be loaded into a flash cart, ESE ram, MEGA ram (or
equivalent) and played on your MSX. 


Compiling pcmenc.exe:
---------------------

To compile pcmenc.exe using gnu gxx, type:

   gxx pcmenc.cpp resample.c -o pcmenc.exe -O2 -Wall

or:

   g++ pcmenc.cpp resample.c -o pcmenc.exe -O2 -Wall

The encoder is only tested on little endian machines but is prepared to
be compiled on big endian machines as well.

````
