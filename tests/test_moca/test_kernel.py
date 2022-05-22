from itertools import product
from collections import Counter

import numpy as np
import numpy.testing as npt
import pytest

from smol.constants import kB
from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.moca.ensemble.semigrand import SemiGrandEnsemble
from smol.moca.sampler.bias import FugacityBias, SquarechargeBias
from smol.moca.sampler.kernel import (
    ALL_MCUSHERS,
    Metropolis,
    Multitry,
    StepTrace,
    ThermalKernel,
    Trace,
    UniformlyRandom,
)
from smol.moca.sampler.mcusher import Flip, Swap, Tableflip
from smol.moca.utils.math_utils import comb
from smol.moca.utils.occu_utils import (get_dim_ids_table, occu_to_species_n,
                                        get_dim_ids_by_sublattice)
from tests.utils import gen_random_occupancy, gen_random_neutral_occupancy

from pymatgen.core import Structure, Lattice

kernels = [UniformlyRandom, Metropolis, Multitry]
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


@pytest.fixture
def multitry_kernel():
    prim = Structure(Lattice.cubic(2.0),
                     [{"Li+": 2 / 3, "Zr4+": 1 / 6, "Mn3+": 1 / 6},
                      {"O2-": 5 / 6, "F-": 1 / 6}],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])
    space = ClusterSubspace.from_cutoffs(prim, {2: 1.0})
    coefs = np.zeros(space.num_corr_functions)
    ce = ClusterExpansion(space, coefs)
    ensemble = SemiGrandEnsemble.from_cluster_expansion(ce, np.diag([3, 1, 1]),
                                                        chemical_potentials=
                                                        {"Li+": 0, "Zr4+": 0,
                                                         "Mn3+": 0, "O2-": 0,
                                                         "F-": 0})
    kernel = Multitry(ensemble, "tableflip", 1E8, k=3,
                      step_kwargs={
                          "optimize_basis": True,  # Make table ergodic.
                          "table_ergodic": True,
                          "swap_weight": 0.2
                      }
                      )
    assert np.isclose(kernel.mcusher.swap_weight, 0.2)
    return kernel


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


def test_temperature_setter(single_canonical_ensemble):
    metropolis_kernel = Metropolis(
        single_canonical_ensemble, step_type="flip", temperature=500
    )
    assert metropolis_kernel.beta == 1 / (kB * metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)


def test_multitry(multitry_kernel):
    def get_n(occu, dim_ids_table):
        return occu_to_species_n(occu, 5, dim_ids_table)

    def get_hash(a):
        return tuple(a.tolist())

    def get_n_states(n, dim_ids):
        assert len(n) == 5
        c = 1
        for dim_id in dim_ids:
            assert n[dim_id].sum() == 3
            n_sl = 3
            for d in dim_id:
                c = c * comb(n_sl, n[d])
                n_sl = n_sl - n[d]
        return c

    sublattices = multitry_kernel.mcusher.sublattices
    dim_ids_table = get_dim_ids_table(sublattices, active_only=True)
    bits = [sublatt.species for sublatt in sublattices]
    dim_ids = get_dim_ids_by_sublattice(bits)
    rand_occu_lmtpo = gen_random_neutral_occupancy(sublattices)
    occu = rand_occu_lmtpo.copy()
    bias = SquarechargeBias(sublattices)
    o_counter = Counter()
    n_counter = Counter()

    # Uniformly random kernel.
    tableflip = multitry_kernel.mcusher
    # print("Sublattices:", tableflip.sublattices)
    # print("Flip table:", tableflip.flip_table)
    l = 100000
    # l = 100
    for i in range(l):
        assert bias.compute_bias(occu) == 0
        n = get_n(occu, dim_ids_table)
        n_counter[get_hash(n)] += 1
        o_counter[get_hash(occu)] += 1

        trace = multitry_kernel.single_step(occu)
        # print("occu:", occu)
        # print("n:", n)
        # print("step:", step)
        occu = trace.occupancy.copy()  # Update.
        # print("occu_next:", occu_next)
        # print("n_next:", get_n(occu_next))

    # When finished, see if distribution is correct.
    assert len(n_counter) == 3
    # print("n_counter:", n_counter)
    n_occus = []
    for n_hash in n_counter.keys():
        n = np.array(n_hash, dtype=int)
        n_occus.append(get_n_states(n, dim_ids))
    n_occus = np.array(n_occus)
    assert len(o_counter) == sum(n_occus)
    o_count_av = l / sum(n_occus)

    n_counts = np.array(list(n_counter.values()))
    r_counts = n_counts / n_counts.sum()
    r_occus = n_occus / n_occus.sum()
    npt.assert_allclose(r_counts, r_occus, atol=0.1)
    npt.assert_allclose(np.array(list(o_counter.values())) / o_count_av,
                        1, atol=0.1)
    # print("n_counts:", n_counts)
    # print("r_counts:", r_counts)
    # print("r_occus:", r_occus)
    # print("occupancies:", o_counter)
    # assert False
    # Read numerical values, see if they are acceptable.
