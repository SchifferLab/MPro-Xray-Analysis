import os, sys

import numpy as np

import argparse
import csv

import operator

from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException

from schrodinger import structure
from schrodinger.structutils import rmsd
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.job import jobcontrol, launcher
from schrodinger.utils import subprocess, cmdline, log

import schrodinger.application.desmond.packages.topo as topo
import schrodinger.application.desmond.packages.traj as traj

logger = log.get_output_logger(__file__)


class Superimposer:
    def __init__(self):
        self._rot = None
        self._tran = None

    def set(self, reference_coords, coords):
        if coords is None or reference_coords is None:
            raise Exception("Invalid coordinates set.")

        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")

        self._calcRotTran(reference_coords, coords)

    def _calcRotTran(self, reference_coords, coords):
        "Superimpose the coordinate sets."

        # center on centroid
        self.c1 = np.mean(coords, axis=0)
        self.c2 = np.mean(reference_coords, axis=0)

        coords = coords - self.c1
        reference_coords = reference_coords - self.c2

        # correlation matrix
        a = np.dot(np.transpose(coords), reference_coords)

        u, d, vt = np.linalg.svd(a)

        self._rot = np.dot(u, vt)

        # check if we have found a reflection
        if np.linalg.det(self._rot) < 0:
            vt[2] = -vt[2]
            self._rot = np.dot(u, vt)
        self._tran = self.c2 - np.dot(self.c1, self._rot)

    def getRotTran(self):
        "Return rotation matrix and translation vector."
        if self._rot is None:
            raise Exception("Nothing superimposed yet.")
        return self._rot, self._tran

    def apply(self, coords):
        if self._rot is None:
            raise Exception("Nothing superimposed yet.")

        return np.dot(coords, self._rot) + self._tran


class Quaternion:
    def __init__(self):
        self._rot = None
        self._tran = None

    def set(self, reference_coords, coords):
        if coords is None or reference_coords is None:
            raise Exception("Invalid coordinates set.")

        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")

        self.quaternion_rotate(coords, reference_coords)

    def apply(self, coords):
        if self._rot is None:
            raise Exception("Nothing superimposed yet.")

        return np.dot(coords, self._rot) + self._tran

    def quaternion_transform(self, r):
        """
        Get optimal rotation
        note: translation will be zero when the centroids of each molecule are the
        same
        """
        Wt_r = self.makeW(*r).T
        Q_r = self.makeQ(*r)
        rot = Wt_r.dot(Q_r)[:3, :3]
        return rot

    def makeW(self, r1, r2, r3, r4=0):
        """
        matrix involved in quaternion rotation
        """
        W = np.asarray([
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4]])
        return W

    def makeQ(self, r1, r2, r3, r4=0):
        """
        matrix involved in quaternion rotation
        """
        Q = np.asarray([
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4]])
        return Q

    def quaternion_rotate(self, X, Y):
        """
        Calculate the rotation
        Parameters
        ----------
        X : array
            (N,D) matrix, where N is points and D is dimension.
        Y: array
            (N,D) matrix, where N is points and D is dimension.
        Returns
        -------
        rot : matrix
            Rotation matrix (D,D)
        """
        N = X.shape[0]
        W = np.asarray([self.makeW(*Y[k]) for k in range(N)])
        Q = np.asarray([self.makeQ(*X[k]) for k in range(N)])
        Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
        W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
        A = np.sum(Qt_dot_W, axis=0)
        eigen = np.linalg.eigh(A)
        r = eigen[1][:, eigen[0].argmax()]
        self._rot = self.quaternion_transform(r)
        self._tran = np.mean(X, axis=0) - np.dot(np.mean(Y, axis=0), self._rot)


def get_parser():
    script_desc = '''Align all frames of a desmond trajectory using the Kabsch algorithm.
    If not specified otherwise, Calpha coordinates are used to calculate the transformation matrix.
    A separate reference structure can be provided Chain identity is assumed to be conserved between trajectory and reference.
    (e.g. chain A in the input trajectory equals to chain A in the reference structure)
    
    This implementation of the Kabsh algorith differs from similar implementations in the fact that cordinates are ordered by
    sequence identity before calculating the transformation matrix. 
    '''

    parser = argparse.ArgumentParser(description=script_desc)

    parser.add_argument('infile',
                        help='Input structure files',
                        nargs='+')

    parser.add_argument('-ref',
                        help='reference structure',
                        type=str,
                        default='')

    parser.add_argument('-align',
                        help='subset of atoms used to calculate the transformation matrix',
                        type=str,
                        default=' protein and a. CA')

    parser.add_argument('-rmsd',
                        help='output the rmsd in a separate file',
                        action='store_true',
                        default=False)

    parser.add_argument('-rmsd_cutoff',
                        help='Drop all frames with a rmsd above cutoff',
                        type=float,
                        default=9999.0)

    parser.add_argument('-extract',
                        help='subset of atoms to write to outfile',
                        type=str,
                        metavar='string',
                        default=None)

    parser.add_argument('-jobname',
                        help='The job name.',
                        type=str,
                        metavar='string',
                        default='atom_pair_alignment')

    # Job control options:
    cmdline.add_jobcontrol_options(
        parser,
        options=(cmdline.WAIT, cmdline.NOJOBID, cmdline.LOCAL, cmdline.HOST))

    return parser


def submit_under_jobcontrol(args, script_args):
    """
    Submit this script under job control.
    """
    # Launch the backend under job control:
    scriptlauncher = launcher.Launcher(
        script=os.path.realpath(__file__),
        jobname=args.jobname,
        local=args.local,
        prog='pair_align',
        wait=args.wait,
        no_redirect=False, )

    # Add input file to jlaunch input file list:
    for f in args.infile:
        scriptlauncher.addInputFile(os.path.realpath(f))

    # Add script arguments:
    # Since the backend will already be running under job control.
    for arg in ["-JOBID", "-LOCAL"]:
        if arg in script_args:
            script_args.remove(arg)
    scriptlauncher.addScriptArgs(script_args)

    scriptlauncher.launch()


class test_Backend:
    def __init__(self):
        pass

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)

    def _get_atom_index(self, st, asl):
        '''

        :param st: a schrodinger.structure.Structure object
        :param asl: q set of atoms in maestro atom selection language
        :return atm_ndx_dict: a dictonary of type [(atom.chain,atom.resnum)]:pythonic_atom-index
        '''
        atm_ndx_dict = {}
        st_atoms = [a for a in st.atom]
        atm_ndx = np.array(evaluate_asl(st, asl)) - 1
        for i in atm_ndx:
            resid = (st_atoms[i].chain.strip(), st_atoms[i].resnum)
            if resid not in atm_ndx_dict:
                atm_ndx_dict[resid] = []
            atm_ndx_dict[resid].append(i)
        return atm_ndx_dict

    def _check_atom_dict(self, pairs=None):
        '''
        Check whether mobile and reference atoms are of the same length and type
        :param pairs: a list of required residue pairs TODO
        '''
        if pairs is None:
            ref_set = set(['{}:{}'.format(c, r) for c, r in self.ref_dict.keys()])
            for d in self.in_dict_list:
                in_set = set(['{}:{}'.format(c, r) for c, r in d.keys()])
                diff = ref_set.difference(in_set)
                if diff:
                    self.error('Not the same type and number of mobile and reference atoms')
                    self.info(ref_set)
                    self.info(in_set)
                    sys.exit(1)
        else:
            # TODO atom pair implementation
            self.error('Not implemented yet')
            sys.exit(1)
        return

    def _return_atom_coordinates(self, st, ndx_dict, order=None):
        '''
        Return atomic coordinates in a [N,3] numpy.array
        The coordinates will be ordered according to (1) chain (2) resnum (3) atom index
        :param st: a schrodinger.structure.Structure object
        :param ndx_dict: a dictonary of type [(atom.chain,atom.resnum)]:pythonic_atom-index
        :param order: a list of residue ids according to which to order the coordinates
        :return crd_array: atomic coordinates in a [N,3] numpy.array
        '''
        xyz = st.getXYZ()
        if order is None:
            crd_array = np.zeros((np.sum(list(map(len, ndx_dict.values()))), 3))
            i = 0
            for k in sorted(ndx_dict.keys(), key=operator.itemgetter(0, 1)):
                for j in ndx_dict[k]:
                    crd_array[i] = xyz[j]
                    i += 1
        else:
            # TODO atom pair implementation
            self.error('Not implemented yet')
            sys.exit(1)

        return crd_array

    def _mobile_st(self):
        '''
        Return the mobile structure from the set of input structures
        :yield:
        '''
        c = 0
        for i in range(len(self.in_st)):
            for j, st in enumerate(self.in_st[i]):
                atom_dict = self.in_dict_list[c]
                c += 1
                yield i, j, st, atom_dict

    def run(self, args):

        global outmaegz
        self.info('Command: $SCHRODINGER/run {}'.format(subprocess.list2cmdline(sys.argv)))

        self.args = args

        self.backend = jobcontrol.get_backend()
        print (type(self.backend))
        if self.backend:
            self.backend.setJobProgress(description='Initializing')

        self.info('Loading input parameters')

        # Add infile to jlaunch file list:
        infile_total = len(self.args.infile)
        self.info('{} input structures found'.format(infile_total))
        infiles = []
        for f in self.args.infile:
            infiles.append(jobcontrol.get_runtime_path(f))
        if self.args.ref:
            ref_file = jobcontrol.get_runtime_path(self.args.ref)
            self.info('Reference structure: {}'.format(ref_file))
            ref_title = ref_file.split('/')[-1]
            self.ref_st = structure.Structure.read(ref_file)
        elif len(infiles) > 1:
            self.info('Using first structure as reference')
            ref_title = infiles[0].split('/')[-1]
            self.ref_st = structure.Structure.read(infiles[0])
        else:
            self.error('No reference found! Nothing to do here')
            sys.exit(1)

        # get dictionary of atom_indices for the reference structure
        self.ref_dict = self._get_atom_index(self.ref_st, self.args.align)
        # get atom coordinates for the reference structure
        ref_crds = self._return_atom_coordinates(self.ref_st, self.ref_dict)

        # Load mobile structures and  get dictionaries of atom indices
        self.in_st = []
        self.in_dict_list = []
        for i, f in enumerate(infiles):
            self.in_st.append([])
            if f.split('.')[-1] in ['maegz', 'mae']:
                for st in structure.StructureReader(f):
                    self.in_st[i].append(st)
                    self.in_dict_list.append(self._get_atom_index(st, self.args.align))
            else:
                st = structure.Structure.read(f)
                self.in_st[i].append(st)
                self.in_dict_list.append(self._get_atom_index(st, self.args.align))
        # Check whether the reference and mobile atoms have the same length and type
        self._check_atom_dict()
        # TODO The _check_atom_dict call could be reworked to be part of the infile loading loop
        # TODO This way one could filter out faulty structures rather than dropping the all together

        # Prepare structure output writer
        if self.args.extract is not None:
            outmaefile = '{}_aligned-out.mae.gz'.format(self.args.jobname)
            outmaegz = structure.StructureWriter(outmaefile)
        if self.args.rmsd:
            outcsvfile = self.args.jobname + '-out.csv'
            csvfh = open(outcsvfile, 'w')
            csvwriter = csv.writer(csvfh)
            csvwriter.writerow(['N,Reference', 'Mobile', 'RMSD_before', 'RMSD_after', 'RMSD_Calpha'])
        # loop over structures and create alignment
        self.info('performing alignment')
        # Loop over all structures returning the structure (st) and the indices i and J
        # i is the index of the infile , j is the index of the structure/frame
        N = 0
        sup = Superimposer()
        # sup = Quaternion()
        for i, j, st, atom_dict in self._mobile_st():
            self.info('Mobile structure: {}'.format(infiles[i] + '_{}'.format(j)))
            mobile_crds = self._return_atom_coordinates(st, atom_dict)
            # pre_kabsch_rmsd = rmsd.calculate_in_place_rmsd(self.ref_st,evaluate_asl(self.ref_st,self.args.align),st,evaluate_asl(st,self.args.align))
            pre_kabsch_rmsd = np.sqrt(np.mean((ref_crds - mobile_crds) ** 2))
            self.info('Pre alignment rmsd: {}'.format(pre_kabsch_rmsd))
            sup.set(ref_crds, mobile_crds)
            # Apply transformation
            st.setXYZ(sup.apply(st.getXYZ()))
            # post_kabsch_rmsd = rmsd.calculate_in_place_rmsd(self.ref_st,evaluate_asl(self.ref_st,self.args.align),st,evaluate_asl(st,self.args.align))
            post_kabsch_rmsd = np.sqrt(np.mean((ref_crds - sup.apply(mobile_crds)) ** 2))
            self.info('Post alignment rmsd: {}'.format(post_kabsch_rmsd))
            try:
                calpha_rmsd = rmsd.calculate_in_place_rmsd(self.ref_st, evaluate_asl(self.ref_st, 'protein and a. CA'),
                                                           st, evaluate_asl(st, ' protein and a. CA'))
            except Exception as e:
                self.warning('Unable to perform Calpha alignment')
                calpha_rmsd = 0
            if self.args.extract is not None:
                if post_kabsch_rmsd < self.args.rmsd_cutoff:
                    self.info('Appending')
                    outmaegz.append(st.extract(evaluate_asl(st, self.args.extract)))
                else:
                    self.info('Alignment did not converge to rmsd < {}'.format(self.args.rmsd_cutoff))
            if self.args.rmsd:
                csvwriter.writerow(
                    [N, ref_title, infiles[i] + '_{}'.format(j), pre_kabsch_rmsd, post_kabsch_rmsd, calpha_rmsd])
            N += 1
        self.info('All done')
        outfiles = []
        if self.args.extract is not None:
            outmaegz.close()
            outfiles.append(outmaefile)
            self.backend.addOutputFile(outmaefile)
        if self.args.rmsd:
            csvfh.close()
            outfiles.append(outcsvfile)
            self.backend.addOutputFile(outcsvfile)

        if outfiles:
            self.info('Output files:')
            for f in outfiles:
                self.info(f)


if __name__ == '__main__':
    parser = get_parser()

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if not args.infile:
        parser.error('Please specify an input file')
    for f in args.infile:
        if not os.path.isfile(f):
            msg = 'Input file not found: {}'.format(f)
            parser.error(msg)
    # This variable was set by the top-level script:
    JOBHOST = os.getenv('JOBHOST')  # first host (no ncpus)

    if args.nojobid:
        use_jobcontrol = False
    else:
        # We are in startup mode, submit under job control if -HOST was used.
        use_jobcontrol = bool(JOBHOST)

    if use_jobcontrol:
        submit_under_jobcontrol(args, sys.argv[1:])
    else:
        # Run the backend (either job control was not being used, or we are
        # already running under job control.
        app = test_Backend()
        app.run(args)
