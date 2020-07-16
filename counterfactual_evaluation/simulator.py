from abc import ABCMeta, abstractmethod
from inspect import signature

import numpy as np


class BaseModel(object):
    """
    Base class for simulation models.
    """
    def get_params(self, deep=True):
        """
        Get the attribute values .

        Args:
            deep (bool): whether or not to return the parameters of
                nested models.
        Returns:
            dict: The model parameters.

        """
        params = dict()
        sig = signature(self.__init__)
        for param in sig.parameters.values():
            pname = param.name
            value = getattr(self, pname, None)
            params[pname] = value
            if deep and hasattr(value, 'get_params'):
                nested_params = value.get_params()
                for k, v in nested_params.items():
                    nested_pname = ''.join([pname, '__', k])
                    params[nested_pname] = v
        return params

    def set_params(self, **kwargs):
        """
        Set parameters for this model.
        The parameter names must be consistent with attributes initialized
        during the model creation.

        Args:
            **kwargs: Keywords arguments for parameters to be set.
        Returns:
            self: The model with updated parameters.

        """
        params = self.get_params()
        # sort items by keys in order to set the nested model before its params.
        for k, v in sorted(kwargs.items(), key=lambda x: x[0]):
            pname, _, next_pname = k.partition('__')
            if pname not in params:
                raise ValueError(f'Invalid parameter {pname} for model {self}')
            if next_pname and hasattr(params[pname], 'set_params'):
                next_param = {next_pname: v}
                params[pname].set_params(**next_param)
            else:
                setattr(self, pname, v)
                params[pname] = v  # update view dict.
        return self


class ExaminationModel(BaseModel, metaclass=ABCMeta):
    """
    Interface of Examination model.
    """
    @abstractmethod
    def probabilities(self, ranks, context):
        """
        Defines examination probability models.

        Args:
            ranks: The ranks of documents.
            context: The context feature.

        Returns: Probabilities scores.

        """
        raise NotImplementedError()

    @abstractmethod
    def values(self, ranks, context):
        """
        Random variable values based on probabilities.

        Args:
            ranks: The ranks of documents.
            context: The context feature.

        Returns: Samples drawn from probabilities computed by
            `self.probabilities`.

        """
        raise NotImplementedError()


class NoiseModel(BaseModel, metaclass=ABCMeta):
    """
    Interface of Noise models. `NoiseModel` models the perceived relevance
    given the true relevance and other context features.
    """
    @abstractmethod
    def probabilities(self, rel, context):
        """
        Defines the probabilities of the perceived relevance.

        Args:
            rel: The true relevance.
            context: The context features.

        Returns:
            Probability scores.
        """
        raise NotImplementedError()

    @abstractmethod
    def values(self, rel, context):
        """
        Random variable values based on probabilities.

        Args:
            rel: The true relevance.
            context: The context feature.

        Returns: Samples drawn from probabilities computed by
            `self.probabilities`.

        """
        raise NotImplementedError()


class ClickModel(BaseModel, metaclass=ABCMeta):
    """
    Interface of click models. This is the final layer of the generative
    click model. `ClickModel` defines click probabilities depending on the
    interplay of examination and perceived relevance models.
    """
    @abstractmethod
    def probabilities(self, ranks, rel, context):
        """
        Defines the probabilities of the click.

        Args:
            ranks: The presentation ranks of documents.
            rel: The true relevance.
            context: The context features.

        Returns:
            Probability scores.
        """
        raise NotImplementedError()

    @abstractmethod
    def values(self, ranks, rel, context):
        """
        Random variable values based on probabilities.

        Args:
            ranks: The presentation ranks of documents.
            rel: The true relevance.
            context: The context features.

        Returns: Samples drawn from probabilities computed by
            `self.probabilities`.

        """
        raise NotImplementedError()


class PositionExamination(ExaminationModel):
    """
    Context-free position-based examination model. Specifically,
    P(Examination = 1 | rank = k) = (1 / k) ^ eta.
    """
    def __init__(self, eta, trunc_at_rank=None):
        """

        Args:
            eta (float): The decay factor of position bias.
            trunc_at_rank (int): The lowest rank any user will examine,
                modeling the selection bias.
        """
        self.eta = eta
        self.trunc_at_rank = trunc_at_rank

    def probabilities(self, ranks, context=None):
        """
        Computes examination probabilities given ranks.

        Args:
            ranks (int, numpy.array): The ranks of documents.
            context: Not used.

        Returns:
            float, numpy.array: The examination probabilities.

        """
        probs = np.power(1 / ranks, self.eta)
        if self.trunc_at_rank:
            probs = np.where(ranks <= self.trunc_at_rank, probs, 0)
        return probs

    def values(self, ranks, context=None):
        """
        Draws examination samples for given ranks.

        Args:
            ranks (int, numpy.array): The ranks of documents.
            context: Not used.

        Returns:
            int, numpy.array: The binary examination samples drawn.

        """
        return np.random.binomial(
            1, self.probabilities(ranks, self.trunc_at_rank))


class ContextualPositionExamination(PositionExamination):
    """
    Contextual position-based examination model. Specifically,
    P(Examination = 1 | rank = k) = (1 / k) ^ max(w * x + 1, 0).
    """
    def __init__(self, eta, nfeatures, trunc_at_rank=None, seed=None):
        """

        Args:
            eta (float): The decay factor of position bias.
            nfeatures (int): The dimension of the context feature vector.
            trunc_at_rank (int): The lowest rank any user will examine,
                modeling the selection bias.
            seed (None, int): The numpy random seed.
        """
        super().__init__(eta, trunc_at_rank)
        if seed:
            np.random.seed(seed)
        weights = np.random.uniform(-eta, eta, nfeatures)
        self.weights = weights - np.mean(weights)
        self.trunc_at_rank = trunc_at_rank

    def probabilities(self, ranks, context=None):
        """
        Computes examination probabilities given ranks.

        Args:
            ranks (int, numpy.array): The ranks of documents.
            context: Not used.

        Returns:
            float, numpy.array: The examination probabilities.

        """
        if context is None:
            return super().probabilities(ranks)
        linear_mul = np.dot(context, self.weights) + 1
        factors = np.maximum(linear_mul, 0)
        probs = np.power(1 / ranks, factors)
        if self.trunc_at_rank:
            probs = np.where(ranks <= self.trunc_at_rank, probs, 0)
        return probs


class ClickNoise(NoiseModel):
    """
    Context-free click noise model. It specifies the perceived relevance
    model for a given query-document pair.
    """
    def __init__(self, epsilons):
        """
        Initializes epsilon lookup table for different relevance scores.

        Args:
            epsilons (dict): The mapping of document relevance to the click
            probabilities given the document is examined.
        """
        self.epsilons = epsilons

    def probabilities(self, rel, context=None):
        """
        Computes the perceived relevance probabilities given the true
        relevance `rel`.
        Args:
            rel (int, numpy.array): The true relevance of
                documents.
            context: Not used.

        Returns:
            float, numpy.array: The click probabilities.

        """
        if isinstance(rel, int):
            rel = np.array(rel)
        return np.vectorize(self.epsilons.get)(rel)

    def values(self, rel, context=None):
        return np.random.binomial(1, self.probabilities(rel))


class PositionBasedClickModel(ClickModel):
    """
    The Position-Based Click Model.
    """

    def __init__(self, examination, noise):
        """

        Args:
            examination (ExaminationModel): The examination model.
            noise (NoiseModel): The noise model. It incorporate noise with
                true relevance to generate perceived relevance.
        """
        self.examination = examination
        self.noise = noise

    def probabilities(self, ranks, rel, context):
        exam_prob = self.examination.probabilities(ranks, context)
        rel_prob = self.noise.probabilities(rel, context)
        return exam_prob * rel_prob

    def values(self, ranks, rel, context):
        return np.random.binomial(
            1, self.probabilities(ranks, rel, context))
