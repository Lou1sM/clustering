for i in {0..4}; do
    for j in {0..3}; do
        python min_ex.py --pretrain_only --pretrain_epochs 20 --seed $(($j + 4*$i)) &
    done
    python min_ex.py --pretrain_only --pretrain_epochs 20 --seed $((4*($i+1)))
done

