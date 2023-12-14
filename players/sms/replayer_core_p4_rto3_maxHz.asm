; Replayer core to play packed-volume 66288Hz samples generated by pcmenc
;
; pcmenc should use the following command line arguments:
;
; pcmenc -rto 3 -p 4 -dt1 54 -dt2 54 -dt3 54 file.wav
;
; and optionally -r 16 to split sample into 16KB blocks
;

; There is one channel update per underlying sample.
; We emit three channel updates, as evenly spaced as possible, every 54 cycles, to match an underlying sample at 66288Hz
; The output rate is limited by our time to loop. The data structure is:
; triplet_count dw ; Count of triplets following
; data          <triplets>
; where each pair of triplets is 3 bytes:
; %aaaabbbb %ccccdddd %eeeeffff
; And within each of these, a is the attenuation set to PSG channel 1, then b to channel 2, c to 3, d to 1, e to 2, f to 3.
; Thus the player needs to count triplets (so it knows when to stop), read data from ROM, shuffle it into PSG commands, and output them.
; PLAY_SAMPLE does this the simplest way which costs 54 cycles to perform the slowest section; other sections are padded to make the gaps equal. This means 66288Hz playback for a 3579545Hz CPU.
; PLAY_SAMPLE_2 uses a hypothetical format that has instead an 8-bit counter for 66-byte chunks, which means the check is faster but this ends up only achieving a 5 cycle saving (49 cycles), for 73052Hz. Since the format doesn't exist, it's not very useful.
; PLAY_SAMPLE_3 tries to use 1.5KB of lookup tables to optimise #2 to 38 cycles, for 94199Hz playback.

;-------------------------------------
; Plays one sample
; HL - points to triplet count followed by data
;-------------------------------------
PLAY_SAMPLE:
  ; the first 16 bits are the "sample count" - as this is for -rto 3, it is the number of outputs divided by 3
  ; e.g. 16KB can hold 10921 sets of three nibbles.
  ld c, (hl)
  inc hl
  ld b, (hl)
  inc hl
  
.macro Delay args cycles
  .if cycles == 16
  neg ; 8
  neg ; 8
  .endif
  .if cycles == 13
  ld a,($0000) ; 13
  .endif
  .if cycles == 12
  jr + ; 12
  +:
  .endif
  .if cycles == 10
  inc ix ; 10
  .endif
  .if cycles == 9
  ld a,i ; 9
  .endif
  .if cycles == 6
  dec de ; 6 - only if de is unused!
  .endif
.endm

PsgLoop:
  ; We unroll x6 because we need to alternate low/high
  ; bc is the number of times we output to all three channels
  ; TODO: could allow only multiples of 3 bytes to make loop middle check unnecessary
  ; TODO: a 16KB chunk can hold 16382 bytes of data (after the 2 bytes header), which 
  ;   means 10921 triplets (wasting half a byte). If we want the header to be 8 bits, 
  ;   it needs to be a count of 64 byte chunks, but our data needs a multiple of 3
  ;   so we round up to 66 byte chunks = 44 triplet chunks, wasting 15 bytes per bank.
  ;   This requires the packer to support this chunky packing.

                            ; 13 loop time
  ld a,(0 << 5) | $90       ;  7
  rld                       ; 18
  out ($7f),a               ; 11 -> 54

  Delay                       16
  ld a,(hl)                 ;  7
  inc hl                    ;  6
  and $f                    ;  7
  or (1 << 5) | $90         ;  7
  out ($7f),a               ; 11 -> 54

  Delay                       12
  dec bc                    ;  6
  ld a,(2 << 5) | $90       ;  7
  rld                       ; 18
  out ($7f),a               ; 11 -> 54

  Delay                        9
  ld a,b                    ;  4
  or c                      ;  4
  ret z                     ;  5
  ld a,(hl)                 ;  7
  and $f                    ;  7
  or (0 << 5) | $90         ;  7
  out ($7f),a               ; 11 -> 54
  
  Delay                        6
  inc hl                    ;  6
  dec bc                    ;  6
  ld a,(1 << 5) | $90       ;  7
  rld                       ; 18
  out ($7f),a               ; 11 -> 54
  
  Delay                       16
  ld a,(hl)                 ;  7
  inc hl                    ;  6
  and $f                    ;  7
  or (2 << 5) | $90         ;  7
  out ($7f),a               ; 11 -> 54

  ld a,b                    ;  4
  or c                      ;  4
  jp nz, PsgLoop            ; 10
  ret

PLAY_SAMPLE_2:
  ; 8-bit counter and more unrolling to reduce loop time to 13 cycles
  ; get triplet count
  ld b,(hl)
  inc hl

.macro PlayHi args channel
  ld a,(channel << 5) | $90 ;  7
  rld                       ; 18
  out ($7f),a               ; 11 -> 36
.endm
.macro PlayLo args channel
  ld a,(hl)                 ;  7
  inc hl                    ;  6
  and $f                    ;  7
  or (channel << 5) | $90   ;  7
  out ($7f),a               ; 11 -> 38
.endm
.macro PlayThreeBytes
  PlayHi 0                  ; 36

  Delay                       11
  PlayLo 1                  ; 38

  Delay                       13
  PlayHi 2                  ; 36

  Delay                       11
  PlayLo 0                  ; 38

  Delay                       13
  PlayHi 1                  ; 36
  
  Delay                       11
  PlayLo 2                  ; 38
.endm

-:.repeat 21
  PlayThreeBytes
  Delay 13
  .endr
  PlayThreeBytes
  djnz -                    ; 13
  ret
  
PLAY_SAMPLE_3:
  ld b,(hl)
  inc hl
  
  ld c,>srl4_table_0 ; 7
  ld d,c      ; 4 ; pre-loop must do the same setup as the end-of-loop

.macro PlayHiTable
  ld e,(hl)   ; 7
  ld a,(de)   ; 7
  out ($7f),a ; 11 -> 25 cycles, add 13 to pad to 38
.endm
  
.macro PlayLoTable
  inc hl      ; 6
  inc d       ; 4
  ld a,(de)   ; 7
  out ($7f),a ; 11 -> 28 cycles, add 10 to pad to 38
.endm
  
-:
              ; 13 for loop
  PlayHiTable
  
  .repeat 21
  Delay 10
  PlayLoTable
  Delay 13
  PlayHiTable
  Delay 10
  PlayLoTable
  Delay 13
  PlayHiTable
  Delay 10
  PlayLoTable
  Delay 13
  PlayHiTable
  .endr
  
  Delay 10
  PlayLoTable
  Delay 13
  PlayHiTable
  Delay 10
  PlayLoTable
  Delay 13
  PlayHiTable

  Delay         6
  inc hl      ; 6
  inc d       ; 4
  ld a,(de)   ; 7
  ld d,c      ; 4 ; Extra operation pre-loop
  out ($7f),a ; 11  -> 38
  
  djnz -      ; 13
  ret
  

; We create some lookup tables for all the possible data values for both low and high nibbles.
; These are 256-byte aligned and ordered to match the outputs for a pair of triplets.
.org $1000
srl4_table_0:
  .repeat 256 index n
  .db (0 << 5) | $90 | n>>4
  .endr
lo4_table_1:
  .repeat 256 index n
  .db (1 << 5) | $90 | (n & $f)
  .endr
srl4_table_2:
  .repeat 256 index n
  .db (2 << 5) | $90 | n>>4
  .endr
lo4_table_0:
  .repeat 256 index n
  .db (0 << 5) | $90 | (n & $f)
  .endr
srl4_table_1:
  .repeat 256 index n
  .db (1 << 5) | $90 | n>>4
  .endr
lo4_table_2:
  .repeat 256 index n
  .db (2 << 5) | $90 | (n & $f)
  .endr
