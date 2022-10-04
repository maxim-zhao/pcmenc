#ifndef ST_H
#define ST_H
/*
 * Sound Tools Library - October 11, 1999
 *
 * Copyright 1999 Chris Bagwell
 *
 * This source code is freely redistributable and may be used for
 * any purpose.  This copyright notice must be maintained.
 * Chris Bagwell And Sundry Contributors are not responsible for
 * the consequences of using this software.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#ifndef __GNUC__
typedef   signed int int32_t;
typedef unsigned int uint32_t;
#endif
typedef int32_t st_sample_t;
typedef uint32_t st_size_t;
typedef int32_t st_ssize_t;
typedef uint32_t st_rate_t;

#define ST_BUFSIZ 8192

#define ST_EFF_CHAN     1               /* Effect can mix channels up/down */
#define ST_EFF_RATE     2               /* Effect can alter data rate */
#define ST_EFF_MCHAN    4               /* Effect can handle multi-channel */
#define ST_EFF_REPORT   8               /* Effect does nothing */

/* Minimum and maximum values a sample can hold. */
#define ST_SAMPLE_MAX 2147483647L
#define ST_SAMPLE_MIN (-ST_SAMPLE_MAX - 1L)
#define ST_SAMPLE_FLOAT_UNSCALE 2147483647.0
#define ST_SAMPLE_FLOAT_SCALE 2147483648.0

#define ST_FLOAT_DDWORD_TO_SAMPLE(d) ((st_sample_t)(d*ST_SAMPLE_FLOAT_UNSCALE))
#define ST_SAMPLE_TO_FLOAT_DDWORD(d) ((double)((double)d/(ST_SAMPLE_FLOAT_SCALE)))

#define M_PI    3.14159265358979323846

#define ST_EOF (-1)
#define ST_SUCCESS (0)

/* util.c */
#define st_report printf
#define st_warn   printf
#define st_fail   printf

static st_sample_t st_gcd(st_sample_t a, st_sample_t b)
{
        if (b == 0)
                return a;
        else
                return st_gcd(b, a % b);
}

typedef struct  st_signalinfo
{
    st_rate_t rate;       /* sampling rate */
    signed char size;     /* word length of data */
    signed char encoding; /* format of sample numbers */
    signed char channels; /* number of sound channels */
    char swap;            /* do byte- or word-swap */
} st_signalinfo_t;

/*
 * Handler structure for each effect.
 */
#define ST_MAX_EFFECT_PRIVSIZE 1000

typedef struct st_effect *eff_t;

typedef struct
{
    const char *name;                  /* effect name */
    unsigned int flags;

    int (*getopts)(eff_t effp, int argc, const char **argv);
    int (*start)(eff_t effp);
    int (*flow)(eff_t effp, st_sample_t *ibuf, st_sample_t *obuf,
                st_size_t *isamp, st_size_t *osamp);
    int (*drain)(eff_t effp, st_sample_t *obuf, st_size_t *osamp);
    int (*stop)(eff_t effp);
} st_effect_t;

struct st_effect
{
    char            *name;          /* effect name */
    struct st_signalinfo ininfo;    /* input signal specifications */
    struct st_signalinfo outinfo;   /* output signal specifications */
    st_effect_t     *h;             /* effects driver */
    st_sample_t     *obuf;          /* output buffer */
    st_size_t       odone, olen;    /* consumed, total length */
    /* The following is a portable trick to align this variable on
     * an 8-byte bounder.  Once this is done, the buffer alloced
     * after it should be align on an 8-byte boundery as well.
     * This lets you cast any structure over the private area
     * without concerns of alignment.
     */
    double priv1;
    char priv[ST_MAX_EFFECT_PRIVSIZE]; /* private area for effect */
};


int st_updateeffect(eff_t effp, st_signalinfo_t *in, st_signalinfo_t *out,
                    int effect_mask);

int st_resample_getopts(eff_t effp, int n, const char **argv);
int st_resample_start(eff_t effp);
int st_resample_flow(eff_t effp, st_sample_t *ibuf, st_sample_t *obuf,
                     st_size_t *isamp, st_size_t *osamp);
int st_resample_drain(eff_t effp, st_sample_t *obuf, st_size_t *osamp);
int st_resample_stop(eff_t effp);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif
