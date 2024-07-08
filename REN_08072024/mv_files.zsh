for dir in ./*;
do 
    cd $dir
    echo "$PWD"
    for file in ./OCR/*;
    do 
        subdir=${file%.txt}
        mkdir -p $subdir/NER
        mv $file $subdir
    done
    cd ../
done
