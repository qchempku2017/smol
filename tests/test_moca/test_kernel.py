from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

from smol.constants import kB
from smol.moca.sampler.bias import FugacityBias
from smol.moca.sampler.kernel import (
    ALL_MCUSHERS,
    Metropolis,
    StepTrace,
    ThermalKernel,
    Trace,
    UniformlyRandom,
    WangLandauImportance
)
from smol.moca.sampler.mcusher import Flip, Swap, Tableflip
from tests.utils import gen_random_occupancy

kernels = [UniformlyRandom, Metropolis]
ushers = ALL_MCUSHERS


@pytest.fixture(params=product(kernels, ushers), scope="module")
def mckernel(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs["temperature"] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    return kernel


@pytest.fixture(params=product(kernels, ushers), scope="module")
def mckernel_bias(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs["temperature"] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    return kernel


@pytest.fixture(params=ushers)
def mckernel_wli(ensemble, request):
    def func1(occu):
        return (np.isclose(occu, 0).sum() / len(occu)) + 0.01  # To avoid zeros.

    def func2(occu):
        data = (np.isclose(occu, 0).sum() / len(occu)) + 0.01
        return np.array([data, data])

    return WangLandauImportance(ensemble, bin_size=1000,
                                min_energy=-10000,
                                max_energy=10000,
                                step_type=request.param,
                                production=False,
                                properties=[("whatever", func1), ("wow", func2)],
                                nwalkers=1)


@pytest.mark.parametrize("step_type, mcusher", [("swap", Swap), ("flip", Flip),
                                                ("tableflip", Tableflip)])
def test_constructor(ensemble, step_type, mcusher):
    kernel = Metropolis(ensemble, step_type=step_type, temperature=500)
    assert isinstance(kernel._usher, mcusher)
    assert isinstance(kernel.trace, StepTrace)
    assert "temperature" in kernel.trace.names
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    assert "bias" in kernel.trace.delta_trace.names


def test_trace(rng):
    trace = Trace(first=np.ones(10), second=np.zeros(10))
    assert all(isinstance(val, np.ndarray) for _, val in trace.items())

    trace.third = rng.random(10)
    assert all(isinstance(val, np.ndarray) for _, val in trace.items())
    names = ["first", "second", "third"]
    assert all(name in names for name in trace.names)

    with pytest.raises(TypeError):
        trace.fourth = "blabla"
        Trace(one=np.zeros(40), two=66)

    steptrace = StepTrace(one=np.zeros(10))
    assert isinstance(steptrace.delta_trace, Trace)
    with pytest.raises(ValueError):
        steptrace.delta_trace = np.ones(10)
        StepTrace(delta_trace=np.ones(10))

    # check saving
    assert trace.as_dict() == trace.__dict__
    steptrace_d = steptrace.__dict__.copy()
    steptrace_d["delta_trace"] = steptrace_d["delta_trace"].__dict__.copy()
    assert steptrace.as_dict() == steptrace_d


def test_single_step(mckernel):
    occu_ = gen_random_occupancy(mckernel._usher.sublattices)
    for _ in range(20):
        trace = mckernel.single_step(occu_.copy())
        if trace.accepted:
            assert not np.array_equal(trace.occupancy, occu_)
        else:
            npt.assert_array_equal(trace.occupancy, occu_)


def test_single_step_bias(mckernel_bias):
    occu = gen_random_occupancy(mckernel_bias._usher.sublattices)
    for _ in range(20):
        trace = mckernel_bias.single_step(occu.copy())
        # assert delta bias is there and recorded
        assert isinstance(trace.delta_trace.bias, np.ndarray)
        print(trace.delta_trace.bias)
        assert len(trace.delta_trace.bias.shape) == 0  # 0 dimensional
        if trace.accepted:
            assert not np.array_equal(trace.occupancy, occu)
        else:
            npt.assert_array_equal(trace.occupancy, occu)


def test_single_step_wli(mckernel_wli):
    occu = gen_random_occupancy(mckernel_wli._usher.sublattices)
    n_accept = 0
    for _ in range(200):
        trace = mckernel_wli.single_step(occu.copy())
        assert isinstance(trace.logsuminv_importance, np.ndarray)
        assert isinstance(trace.logsum_whatever, np.ndarray)
        assert isinstance(trace.logsum_wow, np.ndarray)
        assert trace.logsum_whatever.shape == (len(mckernel_wli._energy_levels), )
        assert trace.logsum_wow.shape == (len(mckernel_wli._energy_levels), 2)
        assert trace.logsuminv_importance.shape == (len(mckernel_wli._energy_levels), )
        logsum_invF = trace.logsuminv_importance[~np.isnan(trace.logsuminv_importance)]
        if len(logsum_invF) > 0:
            assert np.all(logsum_invF <= 0)  # logSum(1/F) should typically be negative.
        if trace.accepted:
            assert not np.array_equal(trace.occupancy, occu)
            n_accept += 1
        else:
            npt.assert_array_equal(trace.occupancy, occu)
        assert n_accept > 0


def test_temperature_setter(single_canonical_ensemble):
    metropolis_kernel = Metropolis(
        single_canonical_ensemble, step_type="flip", temperature=500
    )
    assert metropolis_kernel.beta == 1 / (kB * metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)
