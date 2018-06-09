;==============================================================
; WLA-DX banking setup
;==============================================================
.memorymap
defaultslot 0
slotsize $4000
slot 0 $0000
.endme

.rombankmap
bankstotal 1
banksize $4000
banks 1
.endro

.bank 0 slot 0

.sdsctag 1.00,"psgenc test","","Maxim"

.section "PSG init data" free
PSGInit:
.db $9f $bf $df $ff $81 $00 $a1 $00 $00 $c1 $00
PSGInitEnd:
.ends

.org 0

.section "Boot" force
  ; init Z80
  di
  im 1
  ld sp, $dff0
  
  ; Init PSG
  ld hl,PSGInit
  ld bc,(PSGInitEnd-PSGInit)<<8 + $7f
  otir
  
  .define colours
  
  .ifdef colours
DefaultInitialiseVDP:
  ld hl,_Data
  ld b,_End-_Data
  ld c,$bf
  otir
  jp +
  
_Data:
    .db %00100110,$80
    ;    |||||||`- Disable sync
    ;    ||||||`-- Enable extra height modes
    ;    |||||`--- SMS mode instead of SG
    ;    ||||`---- Shift sprites left 8 pixels
    ;    |||`----- Enable line interrupts
    ;    ||`------ Blank leftmost column for scrolling
    ;    |`------- Fix top 2 rows during horizontal scrolling
    ;    `-------- Fix right 8 columns during vertical scrolling
    .db %10000100,$81
    ;     |||| |`- Zoomed sprites -> 16x16 pixels
    ;     |||| `-- Doubled sprites -> 2 tiles per sprite, 8x16
    ;     |||`---- 30 row/240 line mode
    ;     ||`----- 28 row/224 line mode
    ;     |`------ Enable VBlank interrupts
    ;     `------- Enable display
    .db ($7800>>10)   |%11110001,$82
    .db ($7f00>>7)|%10000001,$85
    .db (1<<2)         |%11111011,$86
    .db $0|$f0,$87
    ;    `-------- Border palette colour (sprite palette)
    .db $00,$88
    ;    ``------- Horizontal scroll
    .db $00,$89
    ;    ``------- Vertical scroll
    .db $ff,$8a
    ;    ``------- Line interrupt spacing ($ff to disable)
_End:

+:
  ; set VDP to CRAM
  ld a,$00
  out ($bf),a
  ld a,$c0
  out ($bf),a
  .endif
  
  ; invoke the player
  ld b,1734/16 ; bank count - first number is size in KB or 0 for maximum (for testing, will play garbage at end)
  ld a,1 ; first bank
-:push bc
    ld ($ffff),a
    inc a
    push af
      ld hl,$8000
      call PLAY_SAMPLE
    pop af
  pop bc
  djnz -

  ; loop forever
-:jr -
.ends

.org $66
.section "Pause" force
retn
.ends

.section "player" align 256
.include "replayer_core_p4_rto3_44010Hz.asm"
;.include "replayer_core_packed_rto3_44011Hz.asm"
.ends

