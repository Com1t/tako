#!/bin/bash

python ./server.py 2>&1 > ./server.log &

for i in {1..20};
do
    python ./client.py `expr 34579 + $i` $i 2>&1 > ./client_$i.log &
done

wait
