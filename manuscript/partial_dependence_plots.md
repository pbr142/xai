# Partial Dependence (PD) profiles/plots

PD plots show how the expected value of model predictions behave as a function of a feature. Partial dependence plots were introduced in the context of gradient boosting machines (GBM) by Friedman (2000).

PD plots are useful to understand to XXX

The plotted profiles can also be used to compare differences in model predictions for different subgroups within the data.

A final use of PD profiles is to compare different models with different levels of flexibility:

* If different approaches which allow for different flexibility **agree** in their PD profile then this is an indication that the relationship is well captured and that the more flexible approaches are not overfitting.
* If the different approaches which allow for different flexibility **disagree** in their PD profile then this suggests possible variable transformations that are needed to the simpler approaches to improve their ability to capture the data.

Also, PD plots are useful to understand and compare model behavior at the boundaries of the range of the observed data. This is useful to understand the generalization performance of models when predictions are made for new data that lies at the edge or even outside of the training data distributions. Some models, such as support vector machines, may become very unstable while others, such as random forests, shrink predictions towards the training data average.


## Definition

The PD profile of a model $f()$ for feature $i$ is defined as the expected value of the model when feature $i$ is fixed to some value $z$:
$$
    PD_j(z) = \mathbb{E}_{\mathbf{X}_{-j}}\left[ f(\mathbf{X}_{j|=z})\right]
$$

Empirically, the expectation is calculated as the average value over the observed sample points:
$$
    \hat{PD}_j(z) = \frac{1}{n} \sum_{i=1}^n f(x_{i|j=z})
$$

where $x_{i|j=z}$ denotes the $i$-th sample value with the $j$-th feature set to $z$.




## Implementations

PD plots are available in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.plot_partial_dependence.html#sklearn.inspection.plot_partial_dependence), [DALEX](https://dalex.drwhy.ai/python/) (Biecek 2018), or [PDPbox](https://github.com/SauceCat/PDPbox) (Jiangchun 2018).





