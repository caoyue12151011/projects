Preamble =======================================================================

1. Length units: 1 chunk = 16 blocks, 1 region = 32 chunks = 512 blocks.

2. Coordinates:
	block_coord = floor(float_coord)
	chunk_coord = floor(block_coord/16)
	region_coord = floor(chunk_coord/32) = floor(block_coord/512)

Pre-generate the MC world ======================================================

1. Install the Chunk-Pregenerator mod via CurseForge: 
https://www.curseforge.com/minecraft/mc-mods/chunkpregenerator.

2. The command used: 
pregen start gen radius mytask SQUARE 0 0 140 minecraft:overworld FAST_CHECK_GEN

Notes:
	r = radius in chunks. 
	1 region .mca file ~ 10 MB
	When r=140, time needed on my MacBook Pro: 40 min.

Load the MC world in Python ====================================================

1 region .p file ~ 2.1 MB (with only a 2D map and y-data of ores)
Time needed per region on my MacBook Pro: 30 min.