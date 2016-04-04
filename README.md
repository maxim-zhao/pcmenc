pcmenc
======

This is a clone of [pcmenc](http://www.dvik-joyrex.com/tools.html), a program which generates high-quality sample data plus matching assembly code players for MSX. 
The features I have added are:

* Support for targetting the SN76489 audio chip, as found in:
** Sega Master System
** Sega Game Gear
** Sega Game 1000
** Sega Computer 3000
** ColecoVision
** BBC Micro
** Lots of arcade games and esoteric home computers
* Optimal bank packing
* Player code for regular Z80 chips, targetting popular sampling rate
* Some minor speedups
* A 64-bit build to allow processing longer files (the process consumes huge amounts of RAM)
