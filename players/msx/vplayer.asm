;----------------------------------------------------------------------------
; Copyright (C) 2006 Arturo Ragozini and Daniel Vik
;
;  This software is provided 'as-is', without any express or implied
;  warranty.  In no event will the authors be held liable for any damages
;  arising from the use of this software.
;
;  Permission is granted to anyone to use this software for any purpose,
;  including commercial applications, and to alter it and redistribute it
;  freely, subject to the following restrictions:
;
;  1. The origin of this software must not be misrepresented; you must not
;     claim that you wrote the original software. If you use this software
;     in a product, an acknowledgment in the product documentation would be
;     appreciated but is not required.
;  2. Altered source versions must be plainly marked as such, and must not be
;     misrepresented as being the original software.
;  3. This notice may not be removed or altered from any source distribution.
;----------------------------------------------------------------------------

ExeType         equ 0   ;     0 = .com file
                        ;     1 = ascii8 .rom file

ReplayerType    equ 2   ;     0 =  8.0   kHz replayer 
                        ;     1 =  8.0   kHz replayer v2 (uses less CPU)
                        ;     2 = 11.025 kHz replayer 
                        ;     3 = 22.05  kHz replayer
                        ;     4 = 44.1   kHz replayer

        IF (ExeType == 0)
        
        OUTPUT vplayer.com
		org	100h
        
        ELSE
        
        OUTPUT vplayer.rom
		org	4000h
		dw	"BA",START,0,0,0,0,0,0
        
        ENDIF

;-------------------------------------
; Entry point
;-------------------------------------       
START:
        di
        call    RESET_PSG
        
        IF (ExeType == 0)

        ld  hl, SAMPLE_START + 2
        ld  de, (SAMPLE_START)

        call PLAY_SAMPLE
        ret
        
        ELSE

.REWIND:
        xor a
.LOOP:
        inc a
        ld  (6800h),a

        ld  hl, SAMPLE_START + 2
        ld  de, (SAMPLE_START)

        ex   af,af'
        call PLAY_SAMPLE
        ex   af,af'
        
        cp  (SAMPLE_END - SAMPLE_START + #1FFF)/#2000
        jr  nz,.LOOP
        jr  .REWIND
        ret
        
        ENDIF
        

;-------------------------------------
; Resets the PSG
;-------------------------------------
RESET_PSG:
        xor     a
        ld      bc,$ffa1
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        inc     a
        out     (c),b
        out     ($a0),a
        ld      b,$bf
        out     (c),b
        ret


;-------------------------------------
; Include the replayer core
;-------------------------------------
        IF (ReplayerType == 0)
            include replayer_core_8000.asm
        ELSE
        IF (ReplayerType == 1)
            include replayer_core_8000v2.asm
        ELSE 
        IF (ReplayerType == 2)
            include replayer_core_11025.asm
        ELSE 
        IF (ReplayerType == 3)
            include replayer_core_22050.asm
        ELSE 
        IF (ReplayerType == 4)
            include replayer_core_44100.asm
        ELSE 
        ENDIF
        ENDIF
        ENDIF
        ENDIF
        ENDIF

;-------------------------------------
; Padding for rom player
;-------------------------------------
        if (ExeType != 0)
        DS (#6000 - $)
        endif


;-------------------------------------
; Sample data
;-------------------------------------
SAMPLE_START:
        incbin "sample.bin"
SAMPLE_END:


;-------------------------------------
; Padding, align rom image to a power of two.
;-------------------------------------
        IF (ExeType != 0)
        
SAMPLE_LENGTH   equ SAMPLE_END - SAMPLE_START

        IF (SAMPLE_LENGTH <= #6000)
        DS (#6000 - SAMPLE_LENGTH)
        ELSE
        IF (SAMPLE_LENGTH <= #E000)
        DS (#E000 - SAMPLE_LENGTH)
        ELSE
        IF (SAMPLE_LENGTH <= #1E000)
        DS (#1E000 - SAMPLE_LENGTH)
        ELSE
        IF (SAMPLE_LENGTH <= #3E000)
        DS (#3E000 - SAMPLE_LENGTH)
        ELSE
        IF (SAMPLE_LENGTH <= #7E000)
        DS (#7E000 - SAMPLE_LENGTH)
        ELSE
        DS (#FE000 - SAMPLE_LENGTH)
        ENDIF
        ENDIF
        ENDIF
        ENDIF
        ENDIF
        
        ENDIF

FINISH:
