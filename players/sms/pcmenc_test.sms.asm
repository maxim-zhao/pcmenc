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

  ; invoke the player
  ld b,256-1 ; bank count - can make it correct if wanted
  ld a,1 ; first bank
-:push bc
    ld ($ffff),a
    inc a
    push af
      ld ix,($8000)
      ld hl,$8002
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

.section "player"
.include "replayer_core_44100.asm"
.ends
