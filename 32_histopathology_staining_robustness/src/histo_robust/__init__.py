"""histo_robust -- Robust histopathology classification under staining variations.

Implements the experiment matrix defined in ``PLAN.md``:

* Axis A -- anchor baseline (ResNet-50, raw RGB, no augmentation)
* Axis B -- colour normalisation (None / Reinhard / Macenko)
* Axis C -- augmentation policy (None / Aug-Geo / Aug-Stain / Aug-Combined)
* Axis D -- normalisation x augmentation interaction
* Axis E -- backbone comparison (ResNet-50 / ConvNeXt-Tiny / Phikon)

Datasets: ``NCT-CRC-HE-100K-NONORM`` (source, in-domain) ->
``CRC-VAL-HE-7K`` (target, out-of-domain, never touched during training).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
