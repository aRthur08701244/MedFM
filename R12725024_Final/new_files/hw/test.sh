for task in chest colon endo
do
    for epoch in {1..40}
    do
        # echo configs/densenet/densenet121_4xb256_in1k-${task}.py
        # echo work_dirs/densenet121_4xb256_in1k-${task}/epoch_${epoch}.pth
        python tools/test.py \
        configs/densenet/densenet121_4xb256_in1k-${task}.py \
        work_dirs/densenet121_4xb256_in1k-${task}/epoch_${epoch}.pth
    done
done
