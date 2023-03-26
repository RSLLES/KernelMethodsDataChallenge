#!/bin/bash
f=./results/README.md
echo "# Results" > $f

for file in ./results/*.md
do
    if [ $file != $f ]
    then
        filename=$(basename "$file")
        filename="${filename%.*}"
        echo "$filename"
        echo "## $filename" >> $f
        cat $file >> $f
        echo "" >> $f
    fi
done

python ./results/add_information.py