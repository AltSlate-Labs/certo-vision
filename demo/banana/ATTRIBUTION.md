Sample images in `samples/` (green, semi_ripe, ripe, overripe) are from the BananaImageBD ripeness
classification dataset by Project-AgML (Hugging Face `Project-AgML/BananaImageBD_ripeness_classification`),
licensed CC BY 4.0, resized to 256 px by `tools/make_banana.py`. They were held out of `calib.npz`,
which was fitted on the first 140 images per level. `not_a_banana.png` is synthetic.
