# ImageRetrieval

Domain-Invariant Similarity Learning for Image Retrieval

## Related Work

- [How to Train a CAT: Learning Canonical Appearance Transformations for Direct Visual Localization Under Illumination Change](https://github.com/utiasSTARS/cat-net)
- [Night-to-Day Image Translation for Retrieval-based Localization](https://github.com/AAnoosheh/ToDayGAN)
- [Unsupervised Image-to-Image Translation Networks](https://github.com/mingyuliutw/unit)
- [Deforming Autoencoders: Unsupervised Disentangling of Shape and Appearance](https://github.com/zhixinshu/DeformingAutoencoders-pytorch)

## Code Organization

This repository is divided into following submodules:

- `datasets`: Contains important `torch.data.Dataset` subclasses
- `models`: Models and everything model-related
- `utils`: Utility functions and classes

## Training

You can take a look at `main_vae.py` or `main_triplet.py` to see how training is performed. In principle, you have to 
instanciate a `utils.config.TrainingConfig`, specifying dataset, model as well as parameters. Then, a `utils.training.Trainer`
can be instanciated with the training config. The training is started by calling `train()` on the `Trainer` object. Training can
be fully resumed from checkpoints which are saved every `save_every` epochs (specified in the `TrainingConfig`).