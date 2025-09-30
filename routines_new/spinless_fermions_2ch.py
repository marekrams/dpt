""" Generator of basic local spingless-fermion operators. """
from __future__ import annotations
from yastn import YastnError, Tensor, Leg
from yastn.operators import meta_operators

NC = ((1, 0), (0, 1))

class SpinlessFermions2ch(meta_operators):
    """ Predefine operators for spinless fermions. """

    def __init__(self, sym='U1xU1', **kwargs):
        r"""
        Standard operators for single fermionic species and 2-dimensional Hilbert space.
        Defines identity, creation, annihilation, and density operators.
        Defines vectors for empty and occupied states, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
            sym : str
                Explicit symmetry to used. Allowed options are :code:`'Z2'`, or :code:`'U1'`.

            **kwargs : any
                Passed to :meth:`yastn.make_config` to change backend,
                default_device or other config parameters.

        Fixes :code:`fermionic` fields in config to :code:`True`.
        """
        if sym not in ('Z2xZ2', 'U1xU1'):
            raise YastnError("For SpinlessFermions sym should be in ('Z2xZ2', 'U1xU1 ').")
        kwargs['fermionic'] = True
        kwargs['sym'] = sym
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')
        self.nch = ((1, 0), (0, 1))

    def space(self, ch=0) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        return Leg(self.config, s=1, t=((0, 0), NC[ch]), D=(1, 1))  # the same for U1xU1 and Z2xZ2

    def I(self, ch=0) -> yastn.Tensor:
        r""" Identity operator. """
        I = Tensor(config=self.config, s=self.s, n=(0, 0))
        I.set_block(ts=((0, 0), (0, 0)), Ds=(1, 1), val=1)
        I.set_block(ts=(NC[ch], NC[ch]), Ds=(1, 1), val=1)
        return I

    def n(self, ch=0) -> yastn.Tensor:
        r""" Particle number operator. """
        n = Tensor(config=self.config, s=self.s, n=(0, 0))
        # n.set_block(ts=(0, 0), Ds=(1, 1), val=0)
        n.set_block(ts=(NC[ch], NC[ch]), Ds=(1, 1), val=1)
        return n

    def vec_n(self, ch=0, val=0) -> yastn.Tensor:
        r""" State with occupation 0 or 1. """
        if val not in (0, 1):
            raise YastnError("Occupation val should be in (0, 1).")
        nv = (0, 0) * (1 - val) + NC[ch] * val
        vec = Tensor(config=self.config, s=(1,), n=nv)
        vec.set_block(ts=(nv,), Ds=(1,), val=1)
        return vec

    def cp(self, ch=0) -> yastn.Tensor:
        r""" Raising operator. """
        cp = Tensor(config=self.config, s=self.s, n=NC[ch])
        cp.set_block(ts=(NC[ch], (0, 0)), Ds=(1, 1), val=1)
        return cp

    def c(self, ch=0) -> yastn.Tensor:
        r""" Lowering operator. """
        n = NC[ch] if self._sym == 'Z2' else tuple(-x for x in NC[ch])
        c = Tensor(config=self.config, s=self.s, n=n)
        c.set_block(ts=((0, 0), NC[ch]), Ds=(1, 1), val=1)
        return c
