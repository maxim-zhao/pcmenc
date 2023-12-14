;----------------------------------------------------------------------------
; Copyright (C) 2006 Arturo Ragozini and Daniel Vik
;
; This software is provided 'as-is', without any express or implied
; warranty. In no event will the authors be held liable for any damages
; arising from the use of this software.
;
; Permission is granted to anyone to use this software for any purpose,
; including commercial applications, and to alter it and redistribute it
; freely, subject to the following restrictions:
;
; 1. The origin of this software must not be misrepresented; you must not
; claim that you wrote the original software. If you use this software
; in a product, an acknowledgment in the product documentation would be
; appreciated but is not required.
; 2. Altered source versions must be plainly marked as such, and must not be
; misrepresented as being the original software.
; 3. This notice may not be removed or altered from any source distribution.
;----------------------------------------------------------------------------

; Modified in 2016 by Maxim for operation on Sega 8-bit consoles (and similar 
; hardware with IO-mapped SN76489 variants)

;
; Replayer core to play RLE encoded 22028Hz samples generated by pcmenc
;
; pcmenc should use the following command line arguments:
;
; pcmenc -p 0 -rto 2 -dt1 157 -dt2 12 -dt3 156 file.wav
;
; and optionally -r to split sample into blocks for rom replayer
;

; There are three channel updates per two underlying samples.
; One channel update is for the odd samples, the other two are
; for the even ones. We try to make these second two as close together 
; as possible, and equalise the spacing to the first. 
; The total loop length is 325 cycles, to match an underlying sample 
; at 22028Hz (3579545 / 325 * 2 = 22028.0).
; This leaves our timings as:
; Calculate sample A              48
; Waste some time                 50
; Emit sample B                   12 -> dt3 = 156 cycles since C
; Calculate sample B              57
; Calculate sample C              57
; Prepare to emit B and C         27
; Waste more time                  4
; Emit sample B                   12 -> dt1 = 157 cycles since A
; Emit sample C                   12 -> dt2 =  12 cycles since B
; Clean up after emitting B and C 10
; Loop                            36

;-------------------------------------
; Plays one sample
; HL - pointes to triplet count followed by data
;-------------------------------------
PLAY_SAMPLE:
  ld de,0
  ld bc,$007f ; so we can use out (c),a
  ; get the triplet count
  ld a, (hl)
  inc hl
  ld ixl, a
  ld a, (hl)
  inc hl
  ld ixh, a
  
PsgLoop:
; Calculate and output channel A volume
  ld a,b          ; 4
  sub $10         ; 7
  jr nc,PsgWaitA  ; 7/12
  ld a,(hl)       ; 7
  inc hl          ; 6
  ld b,a          ; 4
  and $0f         ; 7
  or $90          ; 7 -> 48

  ; We delay 50 cycles (without changing any registers)
  push hl         ; 11
  pop hl          ; 10
  push hl         ; 11
  pop hl          ; 10
  bit 0,a         ;  8 -> 50
  
  ; Then emit it
  out (c),a       ; 12 -> 12
PsgDoneA:

; Calculate channel B volume
  ld a,d          ; 4
  sub $10         ; 7
  jr nc,PsgWaitB  ; 7/12
  ld a,(hl)       ; 7
  inc hl          ; 6
  ld d,a          ; 4
  and $0f         ; 7
  or $b0          ; 7
  ld iyh,a        ; 8 -> 57
PsgDoneB:
  
; Calculate channel C volume
  ld a,e          ; 4
  sub $10         ; 7
  jr nc,PsgWaitC  ; 7/12
  ld a,(hl)       ; 7
  inc hl          ; 6
  ld e,a          ; 4
  and $0f         ; 7
  or $d0          ; 7
  ld iyl,a        ; 8 -> 57
PsgDoneC:

  ; Waste 4 cycles
  nop             ; 4 -> 4

  ; Output channels B and C
  push de         ; 11
    ld d,iyh      ;  8
    ld e,iyl      ;  8
    out (c),d     ; 12 -> 39
    out (c),e     ; 12 -> 12
  pop de          ; 10
  
  ; Decrement length and return if zero
  dec ix          ; 10
  ld a,ixh        ;  8
  or ixl          ;  8
  jp nz,PsgLoop   ; 10 -> 46
  ret
  
PsgWaitA:
  ld b,a      ;  4
  push bc     ; 11
    ld b,3    ;  7
-:  djnz -    ; 13*2+8
  pop bc      ; 10
  ld a,i      ;  9
  jr PsgDoneA ; 12 -> 87
  
PsgWaitB:
  ld d,a      ;  4
  ld a,i      ;  9
  ld a,i      ;  9
  jr PsgDoneB ; 12 -> 34
  
PsgWaitC:
  ld e,a      ;  4
  ld a,i      ;  9
  ld a,i      ;  9
  jr PsgDoneC ; 12 -> 34
