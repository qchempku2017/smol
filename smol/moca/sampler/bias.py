"""Bias term definitions for biased sampling techniques."""
from abc import ABC, abstractmethod
import numpy as np

from ..comp_space import get_oxi_state
from smol.utils import derived_class_factory


class MCMCBias(ABC):
    """Base bias term class.

    Attributes:
        sublattices(List[Sublattice]):
            List of sublattices, including all active
            and inactive sites.
    """

    def __init__(self, all_sublattices, *args, **kwargs):
        """Initialize Basebias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        self.sublattices = all_sublattices

    @abstractmethod
    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return

    @abstractmethod
    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return


class Nullbias(MCMCBias):
    """Null bias, always 0."""

    def __init__(self, all_sublattices, *args, **kwargs):
        """Initialize Nullbias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        super().__init__(all_sublattices, *args, **kwargs)

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return 0

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return 0


class Squarechargebias(MCMCBias):
    """Square charge bias term class, lam * C^2."""

    def __init__(self, all_sublattices, lam=0.5, *args, **kwargs):
        """Initialize Squarechargebias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            lam(Float, optional):
                Lam value in bias term. Should be positive. Default to 0.5.
        """
        super().__init__(all_sublattices, *args, **kwargs)
        self.lam = lam

        if self.lam < 0:
            raise ValueError("Lambda in charge bias can not be negative!")

        self._charge_table = self._build_charge_table()

    def _build_charge_table(self):
        """Build array containing charge of species on each site.

        Rows reperesent sites and columns represent species. Allows
        quick evaluation of charge and charge change from steps.
        """
        num_cols = max(len(s.site_space) for s in self.sublattices)
        num_rows = sum(len(s.sites) for s in self.sublattices)

        table = np.zeros((num_rows, num_cols))
        for s in self.sublattices:
            ordered_cs = [get_oxi_state(sp) for sp in s.site_space]
            table[s.sites, :len(ordered_cs)] = ordered_cs
        return table

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return (self.lam *
                np.sum([self._charge_table[i, o]
                       for i, o in enumerate(occupancy)])**2)

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        occu = occupancy.copy()
        del_c = 0
        for i, sp in step:
            del_c += (self._charge_table[i, sp] -
                      self._charge_table[i, occu[i]])
            occu[i] = sp

        c = np.sum([self._charge_table[i, o]
                   for i, o in enumerate(occupancy)])

        return self.lam * (del_c**2 + 2 * del_c * c)


def mcbias_factory(bias_type, all_sublattices, *args, **kwargs):
    """Get a MCMC bias from string name.

    Args:
        bias_type (str):
            string specyting bias name to instantiate.
        all_sublattices (list of Sublattice):
            list of Sublattices to calculate bias for.
            Must contain all sublattices, including active
            and inactive, otherwise might be unable to
            calculate some type of bias, for example,
            charge bias.
        *args:
            positional args to instatiate a bias term.
        *kwargs:
            Keyword argument to instantiate a bias term.
    """
    return derived_class_factory(bias_type.capitalize(), MCMCBias,
                                 all_sublattices, *args, **kwargs)