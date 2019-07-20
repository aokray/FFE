# FKRFFE
Code Repository for Fair Kernel Regression via Fair Feature Embedding
in Kernel Space (https://arxiv.org/abs/1907.02242)

# Explanation
Fair Feature Embeddings (FFE) are *learned* features in a kernel space,
meaning that by using FFE you are operating in an RKHS and also performing
feature selection simultaneously.

# Usage
FFE is very similar to SkLearn data set transform tools. For example, taking
a data set and transforming it to a fair representation in an RKHS using a
polynomial kernel:

    \# Given a data set "sample", indexes of the unprotected class "unprotected_idxs",
    \# and indexes of the protected class "protected_idxs"
    from _ffe import FFE

    polyFairTransform = FFE(
                        unprotected_idxs,
                        protected_idxs,
                        'polynomial',
                        kernel_params={'degree':4, 'coef0':0.1}
                        )
    fair_data = polyFairTransform(sample)


