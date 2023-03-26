#!/bin/bash
configs=./configs/*.py

for compar_dir in ./results/ ./export/
do
    echo "Configs not found in ${compar_dir}"
    for file in ${configs}
    do
        filename=$(basename "$file" .py)
        res=$(ls ${compar_dir} | grep -q "^${filename}\..*$")
        if [ $? -ne 0 ]
        then
            echo $filename
        fi
    done
done