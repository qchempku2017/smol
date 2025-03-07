"""Utilities to perform groundstate tests. Only valid for solver_test ensemble."""

import numpy as np
import numpy.testing as npt
from pymatgen.core import Species

from smol.capp.generate.groundstate.upper_bound.indices import (
    get_sublattice_indices_by_site,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_variable_values_from_occupancy,
)
from smol.cofe.space.domain import Vacancy
from smol.moca.kernel.mcusher import Swap


def get_random_solver_test_occu(sublattices):
    # Only use this on orig_sublattices.
    # Li+ 3 Mn2+ 3 Mn4+ 2 Ti4+ 1 Vac 3 O2- 9 F- 3
    cation_sublattice = None
    anion_sublattice = None
    for sublattice in sublattices:
        if Species("Li", 1) in sublattice.species:
            cation_sublattice = sublattice
        if Species("O", -2) in sublattice.species:
            anion_sublattice = sublattice

    li_va_sites = np.random.choice(cation_sublattice.sites, size=6, replace=False)
    li_sites = np.random.choice(li_va_sites, size=3, replace=False)
    va_sites = np.setdiff1d(li_va_sites, li_sites)
    mn_ti_sites = np.setdiff1d(cation_sublattice.sites, li_va_sites)
    ti_sites = np.random.choice(mn_ti_sites, size=1, replace=False)
    mn_sites = np.setdiff1d(mn_ti_sites, ti_sites)
    mn2_sites = np.random.choice(mn_sites, size=3, replace=False)
    mn4_sites = np.setdiff1d(mn_sites, mn2_sites)

    o_sites = np.random.choice(anion_sublattice.sites, size=9, replace=False)
    o2_sites = o_sites
    f_sites = np.setdiff1d(anion_sublattice.sites, o_sites)

    li_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Li", 1))
    ]
    mn2_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Mn", 2))
    ]
    mn4_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Mn", 4))
    ]
    ti_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Ti", 4))
    ]
    va_code = cation_sublattice.encoding[cation_sublattice.species.index(Vacancy())]
    o2_code = anion_sublattice.encoding[
        anion_sublattice.species.index(Species("O", -2))
    ]
    f_code = anion_sublattice.encoding[anion_sublattice.species.index(Species("F", -1))]

    occu = np.zeros(24, dtype=np.int32) - 1
    occu[li_sites] = li_code
    occu[mn2_sites] = mn2_code
    occu[mn4_sites] = mn4_code
    occu[ti_sites] = ti_code
    occu[va_sites] = va_code
    occu[o2_sites] = o2_code
    occu[f_sites] = f_code

    assert np.all(occu >= 0)

    return occu.astype(np.int32)


def get_random_variable_values(sublattices):
    # Not always charge balanced.
    num_sites = sum(len(s.sites) for s in sublattices)
    site_sublattice_ids = get_sublattice_indices_by_site(sublattices)

    values = []
    for site_id in range(num_sites):
        sl_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sl_id]
        if len(sublattice.species) > 1 and site_id in sublattice.active_sites:
            site_vals = [0 for _ in range(len(sublattice.species))]
            site_vals[np.random.choice(len(sublattice.species))] = 1
            values.extend(site_vals)

    return np.array(values, dtype=int)


def get_random_neutral_occupancy(
    sublattices, initial_occupancy, canonical=False, force_flip=False
):
    # Guarantee charge balanced.
    ti_code = None
    mn4_code = None
    ti_sites = None
    mn4_sites = None
    for sublattice in sublattices:
        if Species("Ti", 4) in sublattice.species:
            ti_code = sublattice.encoding[sublattice.species.index(Species("Ti", 4))]
            mn4_code = sublattice.encoding[sublattice.species.index(Species("Mn", 4))]
            ti_sites = sublattice.sites[initial_occupancy[sublattice.sites] == ti_code]
            mn4_sites = sublattice.sites[
                initial_occupancy[sublattice.sites] == mn4_code
            ]
    if canonical:
        threshold = 1.1
    elif not force_flip:
        threshold = 0.5
    else:
        threshold = -0.1

    if np.random.random() > threshold:
        # Ti-Mn flip that would not change charge.
        if np.random.random() > 0.5:
            flip = [(np.random.choice(ti_sites), mn4_code)]
        else:
            flip = [(np.random.choice(mn4_sites), ti_code)]
    else:
        swapper = Swap(sublattices)
        flip = swapper.propose_step(initial_occupancy)

    # Apply that flip.
    rand_occu = initial_occupancy.copy()
    for site_id, code in flip:
        rand_occu[site_id] = code
    return rand_occu.astype(np.int32)


def get_random_neutral_variable_values(
    sublattices, initial_occupancy, variable_indices, canonical=False, force_flip=False
):
    # Guarantee charge balanced.
    rand_occu = get_random_neutral_occupancy(
        sublattices, initial_occupancy, canonical=canonical, force_flip=force_flip
    )
    return get_variable_values_from_occupancy(sublattices, rand_occu, variable_indices)


def validate_correlations_from_occupancy(expansion_processor, occupancy):
    # Check whether our interpretation of corr function is correct.
    # Enforce int32.
    occupancy = np.array(occupancy, dtype=np.int32)
    space = expansion_processor.cluster_subspace
    sc_matrix = expansion_processor.supercell_matrix
    mappings = space.supercell_orbit_mappings(sc_matrix)

    corr = np.zeros(space.num_corr_functions)
    corr[0] = 1
    for orbit, mapping in zip(space.orbits, mappings):
        # Use un-flatten version now for easier access.
        n = orbit.bit_id
        corr_tensors = orbit.correlation_tensors
        n_bit_combos = corr_tensors.shape[0]  # index of bit combos
        mapping = np.array(mapping, dtype=int)
        n_clusters = mapping.shape[0]  # cluster image index
        for bid in range(n_bit_combos):
            for cid in range(n_clusters):
                cluster_state = occupancy[mapping[cid]].tolist()
                corr[n] += corr_tensors[tuple((bid, *cluster_state))]
            corr[n] /= n_clusters
            n += 1
    npt.assert_array_almost_equal(
        corr * expansion_processor.size,
        expansion_processor.compute_feature_vector(occupancy),
    )


def validate_interactions_from_occupancy(decomposition_processor, occupancy):
    # Check whether our interpretation of corr function is correct.
    orbit_tensors = decomposition_processor._interaction_tensors
    # Enforce int32.
    occupancy = np.array(occupancy, dtype=np.int32)
    space = decomposition_processor.cluster_subspace
    sc_matrix = decomposition_processor.supercell_matrix
    mappings = space.supercell_orbit_mappings(sc_matrix)

    corr = np.zeros(decomposition_processor.cluster_subspace.num_orbits)
    corr[0] = orbit_tensors[0]
    n = 1
    for mapping in mappings:
        # Use un-flatten version now for easier access.
        mapping = np.array(mapping, dtype=int)
        n_clusters = mapping.shape[0]  # cluster image index
        for cid in range(n_clusters):
            cluster_state = occupancy[mapping[cid]].tolist()
            corr[n] += orbit_tensors[n][tuple(cluster_state)]
        corr[n] /= n_clusters
        n += 1
    npt.assert_array_almost_equal(
        corr * decomposition_processor.size,
        decomposition_processor.compute_feature_vector(occupancy),
    )


def evaluate_correlations_from_variable_values(grouped_terms, variable_values):
    # Evaluate correlation functions or interactions from variable values.
    variable_values = np.array(variable_values, dtype=int)
    corr = []
    for group in grouped_terms:
        f = 0
        for var_inds, corr_factor, _ in group:
            if len(var_inds) == 0:
                f += corr_factor
            else:
                f += corr_factor * np.prod(variable_values[var_inds])
        corr.append(f)
    return np.array(corr)
