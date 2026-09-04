# TRACE dataset layout

Keep the generator split explicit and immutable:

```text
data/
  train/
    real/
    generator_a/
    generator_b/
  test_in_distribution/
    real/
    generator_a/
    generator_b/
  test_unseen/
    real/
    generator_c/
```

`train.py` treats files under a `real` directory as class `0` and every other source as class `1`. Do not place generator C in `train`; the unseen-generator result is only meaningful if that source is held out completely.

For a code-only smoke test, create toy images with:

```bash
cd backend
python make_demo_data.py --output data/train --count 8
python train.py --data data/train --output weights/trace.pt --epochs 1
```

The generated images are not a valid research dataset and must not be used to report TRACE performance.
