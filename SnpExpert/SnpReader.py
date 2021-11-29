"""
snp class

"""
import numpy as np


class SnpReader:
    """
    class to read touchstone snp files
    """

    def __init__(self,file):
        """
        :param file: str or file-object touchstone file to load
        """

        fid = open(file)
        filename = fid.name
        self.filename = filename
        self.comments = None
        self.version = '1.0'

        self.freq_unit = None
        self.freq_num = None
        self.parameter = None
        self.format = None
        self.res = None
        self.ref = None
        self.noise = None

        self.sp = None
        self.port_num = None
        self.port_names = None
        self.load_file(fid)
        self.z0 = []
        self.comment_variables = None

        fid.close()

    def load_file(self, fid):
        """
        Load the touchstone file into the internal data structures.

        Parameters
        ----------
        fid : file object

        """
        filename = self.filename

        # Check the filename extension.
        # Should be .sNp for Touchstone format V1.0, and .ts for V2
        extension = filename.split('.')[-1].lower()

        if (extension[0] == 's') and (extension[-1] == 'p'):  # sNp
            # check if N is a correct number
            try:
                self.port_num = int(extension[1:-1])
            except (ValueError):
                raise (ValueError(
                    "snp extesion error."))
        elif extension == 'ts':
            pass
        else:
            raise Exception('Filename does not have the expected Touchstone extension (.sNp or .ts)')

        values = []
        while True:
            line = fid.readline()
            if not line:
                break
            # store comments if they precede the option line
            line = line.split('!', 1)
            if len(line) == 2:
                if not self.parameter:
                    if self.comments is None:
                        self.comments = ''
                    self.comments = self.comments + line[1]
                elif line[1].startswith(' Port['):
                    try:
                        port_string, name = line[1].split('=', 1)  # throws ValueError on unpack
                        name = name.strip()
                        garbage, index = port_string.strip().split('[', 1)  # throws ValueError on unpack
                        index = int(index.rstrip(']'))  # throws ValueError on not int-able
                        if index > self.port_num or index <= 0:
                            print(
                                "Port name {0} provided for port number {1} but that's out of range for a file with extension s{2}p".format(
                                    name, index, self.port_num))
                        else:
                            if self.port_names is None:  # Initialize the array at the last minute
                                self.port_names = [''] * self.port_num
                            self.port_names[index - 1] = name
                    except ValueError as e:
                        print("Error extracting port names from line: {0}".format(line))

            # remove the comment (if any) so rest of line can be processed.
            # touchstone files are case-insensitive
            line = line[0].strip().lower()

            # skip the line if there was nothing except comments
            if len(line) == 0:
                continue

            # grab the [version] string
            if line[:9] == '[version]':
                self.version = line.split()[1]
                continue

            # grab the [reference] string
            if line[:11] == '[reference]':
                # The reference impedances can be span after the keyword
                # or on the following line
                self.reference = [float(r) for r in line.split()[2:]]
                if not self.reference:
                    line = fid.readline()
                    self.reference = [float(r) for r in line.split()]
                continue

            # grab the [Number of Ports] string
            if line[:17] == '[number of ports]':
                self.port_num = int(line.split()[-1])
                continue

            # grab the [Number of Frequencies] string
            if line[:23] == '[number of frequencies]':
                self.freq_num = line.split()[-1]
                continue

            # skip the [Network Data] keyword
            if line[:14] == '[network data]':
                continue

            # skip the [End] keyword
            if line[:5] == '[end]':
                continue

            # the option line
            if line[0] == '#':
                toks = line[1:].strip().split()
                # fill the option line with the missing defaults
                toks.extend(['ghz', 's', 'ma', 'r', '50'][len(toks):])
                self.freq_unit = toks[0]
                self.parameter = toks[1]
                self.format = toks[2]
                self.res = toks[4]
                if self.freq_unit not in ['hz', 'khz', 'mhz', 'ghz']:
                    print('ERROR: illegal frequency_unit [%s]', self.freq_unit)
                    # TODO: Raise
                if self.parameter not in 'syzgh':
                    print('ERROR: illegal parameter value [%s]', self.parameter)
                    # TODO: Raise
                if self.format not in ['ma', 'db', 'ri']:
                    print('ERROR: illegal format value [%s]', self.format)
                    # TODO: Raise

                continue

            # collect all values without taking care of there meaning
            # we're separating them later
            values.extend([float(v) for v in line.split()])

        # let's do some post-processing to the read values
        # for s2p parameters there may be noise parameters in the value list
        values = np.asarray(values)
        if self.port_num == 2:
            # the first frequency value that is smaller than the last one is the
            # indicator for the start of the noise section
            # each set of the s-parameter section is 9 values long
            pos = np.where(np.sign(np.diff(values[::9])) == -1)
            if len(pos[0]) != 0:
                # we have noise data in the values
                pos = pos[0][0] + 1  # add 1 because diff reduced it by 1
                noise_values = values[pos * 9:]
                values = values[:pos * 9]
                self.noise = noise_values.reshape((-1, 5))

        if len(values) % (1 + 2 * (self.port_num) ** 2) != 0:
            # incomplete data line / matrix found
            raise AssertionError

        # reshape the values to match the rank
        self.sp = values.reshape((-1, 1 + 2 * self.port_num ** 2))
        # multiplier from the frequency unit
        self.freq_mult = {'hz': 1.0, 'khz': 1e3,
                               'mhz': 1e6, 'ghz': 1e9}.get(self.freq_unit)
        # set the reference to the resistance value if no [reference] is provided
        if not self.ref:
            self.ref = [self.res] * self.port_num

    def get_format(self,format="ri"):
        """
        Returns the file format string used for the given format.

        This is useful to get some information.

        Returns
        -------
        format : string

        """
        if format == 'orig':
            frequency = self.freq_unit
            format = self.format
        else:
            frequency = 'hz'
        return "%s %s %s r %s" % (frequency, self.parameter,
                                  format, self.res)

    def get_sparameter_names(self, format="ri"):
        """
        Generate a list of column names for the s-parameter data.
        The names are different for each format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db, orig (where orig refers to one of the three others)

        Returns
        -------
        names : list
            list of strings

        """
        names = ['frequency']
        if format == 'orig':
            format = self.format
        ext1, ext2 = {'ri':('R','I'),'ma':('M','A'), 'db':('DB','A')}.get(format)
        for r1 in range(self.port_num):
            for r2 in range(self.port_num):
                names.append("S%i%i%s"%(r1+1,r2+1,ext1))
                names.append("S%i%i%s"%(r1+1,r2+1,ext2))
        return names

    def get_sparameter_data(self, format='ri'):
        """
        Get the data of the s-parameter with the given format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db, orig

        supported formats are:
          orig:  unmodified s-parameter data
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitude and angle (degree)

        Returns
        -------
        ret: list
            list of numpy.arrays

        """
        data = {}
        if format == 'orig':
            values = self.sp
        else:
            values = self.sp.copy()
            # use frequency in hz unit
            values[:,0] = values[:,0]*self.freq_mult
            if (self.format == 'db') and (format == 'ma'):
                values[:,1::2] = 10**(values[:,1::2]/20.0)
            elif (self.format == 'db') and (format == 'ri'):
                v_complex = ((10**values[:,1::2]/20.0)
                             * np.exp(1j*np.pi/180 * values[:,2::2]))
                values[:,1::2] = np.real(v_complex)
                values[:,2::2] = np.imag(v_complex)
            elif (self.format == 'ma') and (format == 'db'):
                values[:,1::2] = 20*np.log10(values[:,1::2])
            elif (self.format == 'ma') and (format == 'ri'):
                v_complex = (values[:,1::2] * np.exp(1j*np.pi/180 * values[:,2::2]))
                values[:,1::2] = np.real(v_complex)
                values[:,2::2] = np.imag(v_complex)
            elif (self.format == 'ri') and (format == 'ma'):
                v_complex = np.absolute(values[:,1::2] + 1j* self.sp[:,2::2])
                values[:,1::2] = np.absolute(v_complex)
                values[:,2::2] = np.angle(v_complex)*(180/np.pi)
            elif (self.format == 'ri') and (format == 'db'):
                v_complex = np.absolute(values[:,1::2] + 1j* self.sp[:,2::2])
                values[:,1::2] = 20*np.log10(np.absolute(v_complex))
                values[:,2::2] = np.angle(v_complex)*(180/np.pi)

        for i,n in enumerate(self.get_sparameter_names(format=format)):
            data[n] = values[:,i]

        return data

    def get_sparameter_arrays(self):
        """
        Returns the s-parameters as a tuple of arrays.

        The first element is the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the s-parameters are complex number.

        Returns
        -------
        param : tuple of arrays

        """
        v = self.sp

        if self.format == 'ri':
            v_complex = v[:,1::2] + 1j* v[:,2::2]
        elif self.format == 'ma':
            v_complex = (v[:,1::2] * np.exp(1j*np.pi/180 * v[:,2::2]))
        elif self.format == 'db':
            v_complex = ((10**(v[:,1::2]/20.0)) * np.exp(1j*np.pi/180 * v[:,2::2]))

        return (v[:, 0] * self.freq_mult,
                v_complex.reshape((-1, self.port_num, self.port_num)))
