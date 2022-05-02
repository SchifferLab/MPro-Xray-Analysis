from __future__ import print_function

import os
import json
import tempfile
import argparse
import numpy as np

from schrodinger.application.desmond import ffiostructure
from schrodinger.structure import StructureReader as sr
from schrodinger.structutils import analyze

from schrodinger.job import jobcontrol

FF = "OPLS_2005"
REPULSION = False


def ffassign(mae_file, cms_file, forcefield='OPLS_2005', host='localhost'):
    """
    Assign forcefield parameters
    If forcefield is left blank the default will be used (OPLS2005)
    The alternative is: OPLS3
    """
    template_msj = '''task {
  task = "desmond:auto"
}
build_geometry {
  box = {
     shape = orthorhombic
     size = [10.0 10.0 10.0 ]
     size_type = absolute
  }
  neutralize_system = false
  rezero_system = false
  solvate_system = false
}
assign_forcefield {
  forcefield = "%s"
}''' % (forcefield)

    # NOTE write forcefield assignment file
    msj_file = 'assignff.msj'
    with open(msj_file, 'w') as f:
        f.write(template_msj)

    # NOTE assemble launch command
    cmd = []
    cmd.append('utilities/multisim')
    if host != 'localhost':
        cmd.append('-HOST')
        cmd.append(host)

    cmd.append('-JOBNAME')
    cmd.append('ffasign')
    cmd.append('-m')
    cmd.append('{}'.format(msj_file))
    cmd.append('{}'.format(mae_file))
    cmd.append('-o')
    cmd.append('{}'.format(cms_file))
    job = jobcontrol.launch_job(cmd)
    job.wait()


class FFPARAM:
    __doc__ = """
    Forcefield Parent Class.
    This class parses the forcefield information of a cms structure file.

    Functions:
        fudge(a,b)
            requires: a,b = atomic indices 
            returns f = scaling_factor
            This function determines the scaling factor f
            If a and b are separated by more than 3 bonds f = 1.0 
            If a and b are separated by 3 bonds f = 0.5/0 depending on the force field
            If a and b are separated by less than 3 bonds f = 0.0
        combine(a,b)
            This function gets initiated after reading the forcefield parameters
            It combines the Lennard Jones parameters according to the rules specified in the force field
        vdw(*args)
            requires: Forcefield dependent
        electrostatic(*args)
            requires: Forcefield dependent
    """

    def __init__(self, cms):
        """Get vdw parameters for each atom in the simulation"""

        # Create ffioStructures
        struc = [e for e in ffiostructure.CMSReader(cms)]

        '''
        Merge all but the first structure. 
        Structure 0 represents the full system without forcefield parameters
        This step is quiet time consuming and can properly be optimized.
        For most applications we wont need Solvent/Salt ff parameters.
        However for subsequent calculations we want the original atom indexing,
        which is retored by merging the separate ffiostructures.
        We only have to call this once hence the time spend on this step should not impact calculations too much.
        '''
        if len(struc) == 0:
            raise RuntimeError("Structure Object is empty")
        elif len(struc) == 1:
            raise RuntimeError("No Forcefield paramaters available")
        elif len(struc) == 2:
            ffst = struc[1]
        else:
            ffst = struc[1]
            for i in range(2, len(struc)):
                ffst = ffiostructure.merge_ct(ffst, struc[i])

        self.comb_rule = ffst.ffio.property['s_ffio_comb_rule']
        self.ff = ffst.ffio.property['s_ffio_name']

        # Some additional safety checks:
        if struc[0].atom_total != ffst.atom_total:
            raise RuntimeError("Not all atoms are represented in the ffiostructure object")

        atom_type = {}
        for e in ffst.ffio.vdwtype:
            atom_type[e.property['s_ffio_name'].strip()] = [e.property['r_ffio_c1'], e.property['r_ffio_c2']]

        # Storage array row = atom(i) columns: sigma,epsilon,q
        self.nbparams = np.zeros((struc[0].atom_total, 3))

        # Get the vdw parameters for all atoms (Here we convert atom numbers to pythonic indices)
        for s in range(len(ffst.ffio.site)):
            seq = list(map(float, atom_type[ffst.ffio.site[s + 1].property['s_ffio_vdwtype']]))
            seq.append(float(ffst.ffio.site[s + 1].property['r_ffio_charge']))
            self.nbparams[s] = seq

        # Get exclusion list
        self.exclusion = {}
        for e in ffst.ffio.exclusion:
            ep = e.property
            if ep['i_ffio_ai'] - 1 not in self.exclusion:
                self.exclusion[ep['i_ffio_ai'] - 1] = [ep['i_ffio_aj'] - 1]
            else:
                self.exclusion[ep['i_ffio_ai'] - 1].append(ep['i_ffio_aj'] - 1)

        # Check for 1-4 scaling
        self.scaling_factor = 0.0
        if self.ff == 'OPLS_2005':
            # Opls2005 uses a fudge factor for 1_4 nonbonded interactions
            self.scaling_factor = 0.5
            self.pairs = {}
            for p in ffst.ffio.pair:
                pp = p.property
                if pp['i_ffio_ai'] - 1 not in self.pairs:
                    self.pairs[pp['i_ffio_ai'] - 1] = [pp['i_ffio_aj'] - 1]
                else:
                    self.pairs[pp['i_ffio_ai'] - 1].append(pp['i_ffio_aj'] - 1)

        # set function:
        self._set_fudge()
        self._set_combine()
        self._set_vdw()
        self._set_electrostatic()

    def _set_fudge(self):
        """set 1_4 scaling function"""
        if self.ff == 'OPLS_2005':
            def _fudge(a, b):
                if a in self.pairs:
                    if b in self.pairs[a]:
                        return self.scaling_factor
                elif b in self.pairs:
                    if a in self.pairs[b]:
                        return self.scaling_factor
                if a in self.exclusion:
                    if b in self.exclusion[a]:
                        return 0.0
                elif b in self.exclusion:
                    if a in self.exclusion[b]:
                        return 0.0

                return 1.0
        else:
            raise RuntimeError("Did not recognize forcefield: {0}".format(self.ff))

        self.fudge = _fudge

    def _set_combine(self):
        """Set combination function"""
        if self.comb_rule == 'Geometric':
            def _combine(x, y):
                return np.sqrt(x * y)

        elif self.comb_rule == 'Arithmetic':
            def _combine(x, y):
                return (x + y) / 2
        else:
            raise RuntimeError("Did not recognize combination rule: {0}".format(self.cr))

        self.combine = _combine

    def _set_vdw(self):
        """ Set van der Waals function"""
        if self.ff == 'OPLS_2005':
            def _vdw(sig, eps, r, f):
                return (4 * eps * ((sig ** 12 / r ** 12) - (sig ** 6 / r ** 6))) * f
        if self.ff != 'OPLS_2005':
            raise RuntimeError("Did not recognize forcefield: {0}".format(self.ff))

        self.vdw = _vdw

    def _set_electrostatic(self):
        """
        set electrostatics function
        """

        # Define some constants

        # Coulombs constant:
        # Unit: (J*m)/(C**2) = (J*m)/(C**2)
        Ke = np.float64(8.9875517873681764e9)

        # elementary charge
        # Unit: C
        e = np.float64(1.6021766208e-19)

        # Avogadro constant
        # Unit: 1/mol
        N_A = np.float64(6.022140857e23)

        # Unit Conversions
        J2Kcal = np.float64(0.000239006)
        # meter to angstrom
        m2A = np.float64(1e10)

        const = e ** 2 * N_A * Ke * J2Kcal * m2A

        if self.ff == 'OPLS_2005':
            def _electrostatic(qi, qj, r):
                return (qi * qj) * const / r

        if self.ff != 'OPLS_2005':
            raise RuntimeError("Did not recognize forcefield: {0}".format(self.ff))

        self.electrostatic = _electrostatic


class NB_INTERACTIONS(FFPARAM):
    '''
    Calculate pairwise nonbonded terms from a set of atom indices
    '''

    def __init__(self, cms, asl1=None, asl2=None, cutoff=10.0, verbose=True, repulsion=False):
        # set distance cutoff for nonbonded interactions
        self.cutoff = cutoff
        self.st = next(sr(cms))
        if asl1 is None:
            if asl2 is None:
                raise RuntimeError("No atoms specified")
            else:
                if verbose:
                    print('Only one set of atoms specified comparing set_1 against the rest of the system')
                self.ndx1 = np.array(analyze.evaluate_asl(self.st, asl2)) - 1
                self.ndx2 = np.array([x for x in set(np.arange(self.st.atom_total)).difference(self.ndx1)])
        else:
            self.ndx1 = np.array(analyze.evaluate_asl(self.st, asl1)) - 1

            if asl2 is None:
                if verbose:
                    print('Only one set of atoms specified comparing set_1 against the rest of the system')
                self.ndx2 = np.array([x for x in set(np.arange(self.st.atom_total)).difference(self.ndx1)])
            else:
                self.ndx2 = np.array(analyze.evaluate_asl(self.st, asl2)) - 1

        if len(np.unique(np.concatenate((self.ndx1, self.ndx2)))) < len(self.ndx1) + len(self.ndx2):
            raise RuntimeError('Choose non overlaping atom sets \n{0}\n{1}'.format(asl1, asl2))

        if verbose:
            print('Setting forcefield parameters')
        FFPARAM.__init__(self, cms)

        self.repulsion = repulsion
        if not repulsion:
            if verbose:
                print('Script set to ignore repulsive energy terms\n vdw energies will be cut off at rmin')

        if verbose:
            print('calculating non-bonded potential')
        self.run()

    def _dist(self, a1, a2):
        """measure distance between atom1 and atom2 """
        return np.sqrt(np.sum((a1 - a2) ** 2))

    def run(self):

        self.ljp = np.zeros((len(self.ndx1), len(self.ndx2)))
        self.elec = np.zeros((len(self.ndx1), len(self.ndx2)))

        pos = self.st.getXYZ()
        for n1, i in enumerate(self.ndx1):
            ri = pos[i]
            for n2, j in enumerate(self.ndx2):
                # continue if i==j
                if i == j:
                    continue
                # Calculate distance
                rj = pos[j]
                rij = self._dist(ri, rj)
                # If dist greater than cutoff continue
                if rij > self.cutoff:
                    continue

                # get fudge factor
                fij = self.fudge(i, j)
                # Continue if fudge == 0
                if fij == 0:
                    continue
                # for the opls forcefields a equals sigma, b equals epsilon
                ai, bi, qi = self.nbparams[i]
                aj, bj, qj = self.nbparams[j]
                aij = self.combine(ai, aj)
                bij = self.combine(bi, bj)
                # check wether to flatten potential at rij<rmin
                if not self.repulsion:
                    rmin = 2 ** (1 / 6) * aij
                    if rij < rmin:
                        self.ljp[n1][n2] = -bij
                    else:
                        self.ljp[n1][n2] = self.vdw(aij, bij, rij, fij)
                    self.elec[n1][n2] = self.electrostatic(qi, qj, rij)
                else:
                    self.ljp[n1][n2] = self.vdw(aij, bij, rij, fij)
                    self.elec[n1][n2] = self.electrostatic(qi, qj, rij)

    def per_residue(self):
        """
        Return per residue time series
        """
        electrostatic = {}
        lennardjones = {}

        for n1, i in enumerate(self.ndx1):
            for n2, j in enumerate(self.ndx2):
                # Get Residue id (Resname Resnum Chain)
                a1 = self.st.atom[i + 1]
                a2 = self.st.atom[j + 1]
                resid1 = (a1.resnum, a1.chain)
                resid2 = (a2.resnum, a2.chain)
                if abs(a1.resnum - a2.resnum) <= 1:
                    if a1.chain == a2.chain:
                        continue
                        # Non need to store zeros
                if all([self.elec[n1][n2] == 0, self.ljp[n1][n2] == 0]):
                    continue
                if (resid1, resid2) in electrostatic:
                    electrostatic[(resid1, resid2)] += self.elec[n1][n2]
                else:
                    electrostatic[(resid1, resid2)] = self.elec[n1][n2]
                if (resid1, resid2) in lennardjones:
                    lennardjones[(resid1, resid2)] += self.ljp[n1][n2]
                else:
                    lennardjones[(resid1, resid2)] = self.ljp[n1][n2]

        return electrostatic, lennardjones


def parse_args():
    """
    Argument parser when script is run from commandline
    :return:
    """
    description = '''
    Calculate nonbonded interactions between atom groups.
    In the background this will run the desmond systembuilder to assign forcefield parameters.
    Because tof the limitations with the OPLS forcefield, OPLS3 is proprietary, we use the OPLS2005 forcefiled
    to assign forcefield parameters.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infiles',
                        type=str,
                        help='Input structure. Should be a "complete structure" without missing atoms.')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='xray_nonbonded',
                        help='Outfile prefix')

    return parser.parse_args()


def main(outname, mae_file):
    cwd = os.path.abspath('./')
    outfile = outname+'.json'
    mae_file = os.path.abspath(mae_file)
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        cms_file = 'tmp_ffasign-out.cms'
        # NOTE assign forcefield parameters using default forcefiled (opls2005)
        ffassign(mae_file, cms_file)
        asl1 = 'c. A or c. B'
        asl2 = '(c. C or c. D)'
        # NOTE calculate nonbonded interactions
        nonbonded = NB_INTERACTIONS(cms_file, asl1=asl1, asl2=asl2, repulsion=REPULSION)

        outdict = {
            'parameters': {'asl1': asl1, 'asl2': asl2, 'forcefield': FF, 'infile': mae_file, 'repulsive_forces': REPULSION},
            'data': {}, 'data_perRes': {}}
        for n, i in enumerate(nonbonded.ndx1):
            for m, j in enumerate(nonbonded.ndx2):
                # Check whether potential has been calculated
                if nonbonded.ljp[n][m] < 0.:
                    outdict['data']['{}:{}'.format(i + 1, j + 1)] = nonbonded.ljp[n][m]
        PR_elec, PR_ljp = nonbonded.per_residue()
        for pair, val in PR_ljp.items():
            r1 = ':'.join(map(str, pair[0]))
            r2 = ':'.join(map(str, pair[1]))
            outdict['data_perRes']['{}-{}'.format(r1, r2)] = val
    os.chdir(cwd)

    with open(outfile, 'w') as f:
        json.dump(outdict, f)






if __name__ == '__main__':
    args = parse_args()
    main(args.infiles, args.prefix)
