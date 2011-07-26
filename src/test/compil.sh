#!/bin/bash

W=$1

for i in element graph node edge stream common id iterator graph_bfs algorithm_unweighted_eccentricity algorithm_diameter stream_dgs; do
	gcc -g -c ../gs_${i}.c -I.. `pkg-config --cflags glib-2.0`
	if [ $? -gt 0 ]; then
		echo "*** ERROR ***"
		exit 1
	fi
done

gcc *.o gs_${W}_test.c -I.. `pkg-config --cflags --libs glib-2.0` -o $W -g
