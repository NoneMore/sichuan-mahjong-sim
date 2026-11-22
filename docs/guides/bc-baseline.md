# BC baseline training recipe

Use Suit/Rank ResNet to initialize RL self-play. The flat MLP is the BC
comparison model.

Start with the frozen prepared cache of 50,000 `HeuristicPolicy` games
(game seeds 0–49,999). Verify that its cache ID is
`6ab386e2ee421fd9fa0c6f733ca6300d1e7f9a9f9317ec53521325be847e97b7`.
For setup and cache preparation, follow the [BC pipeline](bc-pipeline.md).

```sh
for seed in 7 17 29; do
  python -m mahjong_bc train --dataset ../bc-cache --output "../bc-resnet-${seed}" \
    --architecture suit-rank-resnet --state-hidden 256 --scorer-hidden 128 --activation relu \
    --seed "$seed" --epochs 50 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.01 \
    --device cuda --data-storage ram --ram-limit-mib 14336 \
    --batch-transfer cuda-prefetch --pinned-limit-mib 64
done
```

The recipe uses AdamW with betas 0.9 and 0.999 and a fixed learning rate. Use each
run's `best.pt`, selected by minimum mean validation NLL. For the MLP control,
use `--architecture mlp` and fresh output directories.
