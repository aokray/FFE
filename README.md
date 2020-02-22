# Fair Feature Embeddings (FFE)
Code Repository for Fair Kernel Regression via Fair Feature Embeddings
in Kernel Space (https://arxiv.org/abs/1907.02242), accepted at ICTAI'19 - if you use this code, please cite our paper.

All data used in our paper can be found here: https://uwyomachinelearning.github.io/.

# Explanation
Fair Feature Embeddings (FFE) are *learned* features in a kernel space, and the
transformed data can be used on standard methods. Note: when using FFE, you are
operating in an RKHS, so you do not need to use the kernelized version of
the method of your choosing, because the features are already in the RKHS.

# Usage
FFE is very similar to SkLearn data set transform tools. For example, taking
a data set and transforming it to a fair representation in an RKHS using a
polynomial kernel:

```python
# Given a data set "sample", indexes of the unprotected class "unprotected_idxs",
# and indexes of the protected class "protected_idxs"
from _ffe import FFE

polyFairTransform = FFE(
    unprotected_idxs,
    protected_idxs,
    "polynomial",
    kernel_params={"degree": 4, "coef0": 0.1},
)
fair_data = polyFairTransform.transform(sample)
```

It's at this point you can use the transformed data ```fair_data``` to
learn a base model; In our case, this is Ridge regression.

