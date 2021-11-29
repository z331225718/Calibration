import numpy as np


class Snp(object):

    def __init__(self, file: str = None, **kwargs) -> None:
        self.port_names = None

        self.deembed = None
        self.noise = None
        self.noise_freq = None
        self._z0 = np.array(50, dtype=complex)
        if file is not None:
            self.read_touchstone(file)



    def read_touchstone(self, filename) -> None:
        """
        loads values from a touchstone file.

        The work of this function is done through the
        :class:`~skrf.io.touchstone` class.

        Parameters
        ----------
        filename : str or file-object
            touchstone file name.


        Note
        ----
        Only the scattering parameters format is supported at the moment


        """
        from .SnpReader import SnpReader
        touchstoneFile = SnpReader(filename)

        if touchstoneFile.get_format().split()[1] != 's':
            raise NotImplementedError('only s-parameters supported for now.')

        self.port_names = touchstoneFile.port_names

        # set z0 before s so that y and z can be computed

        self.z0 = complex(touchstoneFile.res)

        f, self.s = touchstoneFile.get_sparameter_arrays()  # note: freq in Hz
        self.frequency = f
        self.freq_unit = touchstoneFile.freq_unit
