"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log
from types import SimpleNamespace

import numpy as np

from smol.constants import kB
from smol.moca.sampler.bias import MCBias, mcbias_factory
from smol.moca.sampler.mcusher import MCUsher, mcusher_factory
from smol.utils import class_name_from_str, derived_class_factory, get_subclasses

ALL_MCUSHERS = list(get_subclasses(MCUsher).keys())
ALL_BIAS = list(get_subclasses(MCBias).keys())


class Trace(SimpleNamespace):
    """Simple Trace class.

    A Trace is a simple namespace to hold states and values to be recorded
    during MC sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        if not all(isinstance(val, np.ndarray) for val in kwargs.values()):
            raise TypeError("Trace only supports attributes of type ndarray.")
        super().__init__(**kwargs)

    @property
    def names(self):
        """Get all attribute names."""
        return tuple(self.__dict__.keys())

    def items(self):
        """Return generator for (name, attribute)."""
        yield from self.__dict__.items()

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if isinstance(value, (float, int)):
            value = np.array([value])

        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of underlying dictionary."""
        return self.__dict__.copy()


class StepTrace(Trace):
    """StepTrace class.

    Same as Trace above but holds a default "delta_trace" inner trace to hold
    trace values that represent changes from previous values, to be handled
    similarly to delta_features and delta_energy.

    A StepTrace object is set as an MCKernel's attribute to record
    kernel specific values during sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        super().__init__(**kwargs)
        super(Trace, self).__setattr__("delta_trace", Trace())

    @property
    def names(self):
        """Get all field names. Removes delta_trace from field names."""
        return tuple(name for name in super().names if name != "delta_trace")

    def items(self):
        """Return generator for (name, attribute). Skips delta_trace."""
        for name, value in self.__dict__.items():
            if name == "delta_trace":
                continue
            yield name, value

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if name == "delta_trace":
            raise ValueError("Attribute name 'delta_trace' is reserved.")
        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of serializable dictionary."""
        step_trace_d = self.__dict__.copy()
        step_trace_d["delta_trace"] = step_trace_d["delta_trace"].as_dict()
        return step_trace_d


class MCKernel(ABC):
    """Abtract base class for transition kernels.

    A kernel is used to implement a specific MC algorithm used to sample
    the ensemble classes. For an illustrative example of how to derive from this
    and write a specific kernel see the Metropolis kernel.
    """

    # Lists of valid helper classes, set these in derived kernels
    valid_mcushers = None
    valid_bias = None

    def __init__(
        self,
        ensemble,
        step_type,
        *args,
        seed=None,
        step_kwargs=None,
        bias_type=None,
        bias_kwargs=None,
        **kwargs,
    ):
        """Initialize MCKernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            step_kwargs (dict):
                dictionary of keyword arguments to pass to the MCUsher
                constructor.
            seed (int): optional
                non-negative integer to seed the PRNG
            bias_type (str): optional
                name for bias type instance.
            bias_kwargs (dict): optional
                dictionary of keyword arguments to pass to the bias
                constructor.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                Additional keyword arguments to instantiate the Kernel.
        """
        self.natural_params = ensemble.natural_parameters
        self._seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._rng = np.random.default_rng(self._seed)
        self._compute_features = ensemble.compute_feature_vector
        self._feature_change = ensemble.compute_feature_vector_change

        self.trace = StepTrace(accepted=np.array([True]))
        self._usher, self._bias = None, None

        mcusher_name = class_name_from_str(step_type)
        step_kwargs = {} if step_kwargs is None else step_kwargs
        self.mcusher = mcusher_factory(
            mcusher_name,
            ensemble.sublattices,
            *args,
            rng=self._rng,
            **step_kwargs,
        )

        if bias_type is not None:
            bias_name = class_name_from_str(bias_type)
            bias_kwargs = {} if bias_kwargs is None else bias_kwargs
            self.bias = mcbias_factory(
                bias_name,
                ensemble.sublattices,
                rng=self._rng,
                **bias_kwargs,
            )

        # run a initial step to populate trace values
        _ = self.single_step(np.zeros(ensemble.num_sites, dtype=int))

    @property
    def mcusher(self):
        """Get the MCUsher."""
        return self._usher

    @mcusher.setter
    def mcusher(self, usher):
        """Set the MCUsher."""
        if usher.__class__.__name__ not in self.valid_mcushers:
            raise ValueError(f"{type(usher)} is not a valid MCUsher for this kernel.")
        self._usher = usher

    @property
    def seed(self):
        """Get seed for PRNG."""
        return self._seed

    @property
    def bias(self):
        """Get the bias."""
        return self._bias

    @bias.setter
    def bias(self, bias):
        """Set the bias."""
        if bias.__class__.__name__ not in self.valid_bias:
            raise ValueError(f"{type(bias)} is not a valid MCBias for this kernel.")
        if "bias" not in self.trace.delta_trace.names:
            self.trace.delta_trace.bias = np.zeros(1)
        self._bias = bias

    def set_aux_state(self, state, *args, **kwargs):
        """Set the auxiliary state from initial or checkpoint values."""
        self._usher.set_aux_state(state, *args, **kwargs)

    @abstractmethod
    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace: a step trace for states and traced values for a single
                       step
        """
        return self.trace

    def compute_initial_trace(self, occupancy):
        """Compute inital values for sample trace given an occupancy.

        Args:
            occupancy (ndarray):
                Initial occupancy

        Returns:
            Trace
        """
        trace = Trace()
        trace.occupancy = occupancy
        trace.features = self._compute_features(occupancy)
        # set scalar values into shape (1,) array for sampling consistency.
        trace.enthalpy = np.array([np.dot(self.natural_params, trace.features)])
        if self.bias is not None:
            trace.bias = np.array([self.bias.compute_bias(occupancy)])
        trace.accepted = np.array([True])
        return trace


class ThermalKernel(MCKernel):
    """Abtract base class for transition kernels with a set temperature.

    Basically all kernels should derive from this with the exception of those
    for multicanonical sampling and related methods
    """

    def __init__(self, ensemble, step_type, temperature, *args, **kwargs):
        """Initialize ThermalKernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            temperature (float):
                temperature at which the MCMC sampling will be carried out.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                keyword arguments to instantiate the Kernel and MCUsher, Bias,
                etc.
        """
        # hacky for initialization single_step to run
        self.beta = 1.0 / (kB * temperature)
        super().__init__(ensemble, step_type, *args, **kwargs)
        self.temperature = temperature

    @property
    def temperature(self):
        """Get the temperature of kernel."""
        return self.trace.temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self.trace.temperature = np.array(temperature)
        self.beta = 1.0 / (kB * temperature)

    def compute_initial_trace(self, occupancy):
        """Compute inital values for sample trace given occupancy.

        Args:
            occupancy (ndarray):
                Initial occupancy

        Returns:
            Trace
        """
        trace = super().compute_initial_trace(occupancy)
        trace.temperature = self.trace.temperature
        return trace


class UniformlyRandom(MCKernel):
    """A Kernel that accepts all proposed steps.

    This kernel samples the random limit distribution where all states have the
    same probability (corresponding to an infinite temperature). If a
    bias is added then the corresponding distribution will be biased
    accordingly.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self._usher.propose_step(occupancy)
        log_factor = self._usher.compute_log_priori_factor(occupancy, step)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step)
            )
            exponent = self.trace.delta_trace.bias + log_factor
            self.trace.accepted = np.array(
                True
                if self.trace.delta_trace.bias >= 0
                else self.trace.delta_trace.bias > log(self._rng.random())
            )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step)

        self.trace.occupancy = occupancy
        return self.trace


class Metropolis(ThermalKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self._usher.propose_step(occupancy)
        log_factor = self._usher.compute_log_priori_factor(occupancy, step)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step)
            )
            exponent = (
                -self.beta * self.trace.delta_trace.enthalpy
                + self.trace.delta_trace.bias + log_factor
            )
            self.trace.accepted = np.array(
                True if exponent >= 0 else exponent > log(self._rng.random())
            )
        else:
            self.trace.accepted = np.array(
                True
                if self.trace.delta_trace.enthalpy <= 0
                else -self.beta * self.trace.delta_trace.enthalpy
                > log(self._rng.random())  # noqa
            )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step)
        self.trace.occupancy = occupancy

        return self.trace


class Multitry(ThermalKernel):
    """A Multi-try kernel, with lambda=1.

    See J. Am. Stat. Asso. (2000) 95:449, 121-134.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def __init__(self, ensemble, step_type, temperature, *args, k=3, **kwargs):
        """Initialize Multitry kernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            temperature (float):
                temperature at which the MCMC sampling will be carried out.
            k (int): optional
                Number of tries per proposal. Default to 3.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                keyword arguments to instantiate the MCUsher, bias, etc.
        """
        # hacky for initialization single_step to run
        self.k = k
        super(Multitry, self).__init__(ensemble, step_type, temperature,
                                       *args, **kwargs)

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        rng = self._rng
        specs_forth = []
        log_ws_forth = []
        # Make k forth proposals.
        for j in range(self.k):
            step = self._usher.propose_step(occupancy)
            log_p_back, log_p_forth =\
                self._usher.compute_log_priori_factor(occupancy, step,
                                                      return_log_p=True)
            delta_features = self._feature_change(occupancy, step)
            delta_enthalpy = np.array(np.dot(self.natural_params, delta_features))
            if self._bias is not None:
                delta_bias = np.array(self._bias.compute_bias_change(occupancy, step))
            else:
                delta_bias = 0
            specs_forth.append((step, log_p_back, log_p_forth,
                                delta_features, delta_enthalpy,
                                delta_bias))

            log_w = -self.beta * delta_enthalpy + delta_bias + log_p_back
            log_ws_forth.append(log_w)

        # Choose one from k forth proposals.
        w_norm = np.exp(log_ws_forth)
        w_norm = (w_norm / w_norm.sum()
                  if w_norm.sum() != 0
                  else np.ones(len(w_norm)) / len(w_norm))

        ii = rng.choice(self.k, p=w_norm)
        (step_ii, log_p_back_ii, log_p_forth_ii,
         delta_features_ii, delta_enthalpy_ii, delta_bias_ii)\
            = specs_forth[ii]
        occu_ii = occupancy.copy()
        for tup in step_ii:
            occu_ii[tup[0]] = tup[1]

        # Make k-1 back proposals.
        log_ws_back = []
        for j in range(self.k - 1):
            step = self._usher.propose_step(occu_ii)
            log_p_back, log_p_forth =\
                self._usher.compute_log_priori_factor(occu_ii, step,
                                                      return_log_p=True)
            delta_features = self._feature_change(occu_ii, step) + delta_features_ii
            delta_enthalpy = np.array(np.dot(self.natural_params, delta_features))
            if self.bias is not None:
                delta_bias = np.array(self._bias.compute_bias_change(occu_ii, step)
                                      + delta_bias_ii)
            else:
                delta_bias = 0

            log_w = -self.beta * delta_enthalpy + delta_bias + log_p_back
            log_ws_back.append(log_w)
        # Enforce original occupancy as the last back proposal
        log_w_ii = self.beta * delta_enthalpy_ii - delta_bias_ii + log_p_forth_ii
        log_ws_back.append(log_w_ii)

        # Wrapping up proposal and do acceptance.
        self.trace.delta_trace.features = delta_features_ii
        self.trace.delta_trace.enthalpy = delta_enthalpy_ii
        if self._bias is not None:
            self.trace.delta_trace.bias = delta_bias_ii

        # center these for numerical accuracy.
        # This line used to cause problem
        center = min(np.average(log_ws_forth), np.average(log_ws_back))
        log_ws_forth = np.array(log_ws_forth) - center
        log_ws_back = np.array(log_ws_back) - center
        # Adjusted for numerical accuracy.
        ws_forth = np.exp(log_ws_forth)
        ws_back = np.exp(log_ws_back)
        log_sum_ws_forth = (np.log(np.sum(ws_forth)) if np.sum(ws_forth) != 0
                            else -np.inf)
        log_sum_ws_back = (np.log(np.sum(ws_back)) if np.sum(ws_back) != 0
                           else -np.inf)
        exponent = log_sum_ws_forth - log_sum_ws_back
        self.trace.accepted = np.array(True if exponent >= 0
                                       else exponent > log(rng.random()))

        if self.trace.accepted:
            for tup in step_ii:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step_ii)
        self.trace.occupancy = occupancy

        return self.trace


def mckernel_factory(kernel_type, ensemble, step_type, *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            string specifying kernel type to instantiate.
        ensemble (Ensemble)
            an Ensemble object to create the MCMC kernel from.
        step_type (str):
            string specifying the proposal type (i.e. key for MCUsher type)
        *args:
            positional arguments passed to class constructor
        **kwargs:
            keyword arguments passed to class constructor

    Returns:
        MCKernel: instance of derived class.
    """
    kernel_name = class_name_from_str(kernel_type)
    return derived_class_factory(
        kernel_name, MCKernel, ensemble, step_type, *args, **kwargs
    )
