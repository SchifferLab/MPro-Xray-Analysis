"""
Computes "substrate envelope" (van der Waals mask) for a set of ligands.
The ligands are assumed to be aligned appropriately.
"""

import os
import sys
import argparse

import numpy as np
import scipy as sp
from scipy import spatial

import multiprocessing

from schrodinger import structure
from schrodinger.job import jobcontrol, launcher
from schrodinger.utils import cmdline, subprocess, fileutils, log
from schrodinger.structutils import analyze

from schrodinger.analysis.visanalysis import volumedata
from schrodinger.analysis.visanalysis import volumedataio

SCRIPT_FILENAME = os.path.splitext(os.path.basename(__file__))[0] + '.py'
SCRIPT_FILENAME = os.path.join('substrate-envelope', SCRIPT_FILENAME)

logger = log.get_output_logger(__file__)


class MEASURE:
    """
    Python class providing utility functions
    """

    def __init__(self):
        pass

    def distance(self, r1, r2):
        """
        Return euclidean distance r1:r2
        """
        return np.sqrt(np.sum((r1 - r2) ** 2))

    def distance_squared(self, r1, r2):
        """
        Return sqared euclidean distance r1:r2
        """
        return np.sum((r1 - r2) ** 2)

    def distance_matrix(self, a1, a2, tri_matrix=False, diagonal_offset=-1):
        """
        Return distance matrix of shape a1:a2

        If tri_matrix==True:
        All points above the matrix diagonal are set to 0
        In addition to the distance matrix the function will return all valid indices.
        This is usefull when calculating the distance matrix of an array with itself.
        (The diagonal offset is -1 by default this way the trangle doesnt contain the points m[i,j=i]
        """
        if tri_matrix:
            dist_matrix = np.zeros((len(a1), len(a2)))
            for n1, r1 in enumerate(a1):
                for n2, r2 in enumerate(a2):
                    dist_matrix[n1][n2] = self.distance(r1, r2)
            return np.tril(dist_matrix, k=diagonal_offset), np.tril_indices(dist_matrix.shape[0], k=diagonal_offset)

        else:
            dist_matrix = np.zeros((len(a1), len(a2)))
            for n1, r1 in enumerate(a1):
                for n2, r2 in enumerate(a2):
                    dist_matrix[n1][n2] = self.distance(r1, r2)
            return dist_matrix

    def center_of_mass(self, points, mass):
        """
        Return the center of mass  of a n*d array
        Points are weighted by n size mass array
        """
        return np.average(points, axis=0, weights=mass)


class GRID(MEASURE):
    '''
    Discrete!
    '''

    def __init__(self, origin=[0., 0., 0.], length=[1., 1., 1.], spacing=[0.5, 0.5, 0.5]):

        # Init measure
        MEASURE.__init__(self)
        self._spacing = np.array(spacing)
        self._length = np.array(length)
        self._origin = np.array(origin)
        # create grid
        self._grid_points = self._set_grid(origin, length, spacing)
        self._grid_values = np.zeros(len(self._grid_points))
        # Construct Kdist tree from grid points
        self.dist_tree = sp.spatial.cKDTree(self._grid_points)

    def _set_grid(self, origin, length, spacing):
        # Create coordinate vectors
        cv = [np.arange(start=o, stop=o + l*s+s, step=s) for o, l, s in np.vstack((origin, length, spacing)).T]
        # return grid points
        return np.vstack([gp.flatten() for gp in np.meshgrid(*cv)]).T

    def map2grid(self, points, vdw=False):
        """
        Project an array of points onto the grid.
        Points are mapped to grid either based on NearesNeighbour or basedo n their vdw radius.
        If the later is used, the final column must contain the vdw radius for each point

        :param points: A numpy array of N points.
        :param vdw: If true points will be mapped based on their vdw radius.
        """
        # NOTE since our grid will not necesarrily cover the full simulation system we will ignore the outermost layer of grid cells
        xlim = (np.max(self._grid_points[::, 0]), np.min(self._grid_points[::, 0]))
        ylim = (np.max(self._grid_points[::, 1]), np.min(self._grid_points[::, 1]))
        zlim = (np.max(self._grid_points[::, 2]), np.min(self._grid_points[::, 2]))
        if vdw and points.shape[-1] != 4:
            raise RuntimeError('Need to specify vdw radius in input array')
        # get nearest grid point for each point p
        if vdw:
            temp_grid_values = np.zeros(self._grid_values.shape)
            for crd, r in zip(points[::, :3], points[::, 3]):
                ndx = self.dist_tree.query_ball_point(crd, r)
                for i in ndx:
                    # Check wheter any grid points are out of bounds
                    if not any([c in lim for c, lim in zip(self._grid_points[i], [xlim, ylim, zlim])]):
                        temp_grid_values[i] = 1.0
            # Add the vdw surface to the grid
            self._grid_values += temp_grid_values

        else:
            for p in points:
                if self._is_mapable(p):
                    d, i = self.dist_tree.query(p)
                    # Check wheter any grid points are out of bounds
                    if not any([c in lim for c, lim in zip(self._grid_points[i], [xlim, ylim, zlim])]):
                        self._grid_values[i] += 1

    def _is_mapable(self, r):
        """
        Check wether coordinates r are within 1 spacing of the grids borders
        """
        if np.all(r > self._origin - self._spacing):
            if np.all(r < (self._origin + self._length + self._spacing)):
                return True
        return False

    def normalize(self, f=None):
        '''
        normalize the grid values
        If a value is provided for f grid values will be normalized by f
        '''
        if f is None:
            gridsum = np.sum(self._grid_values.flatten())
            self._grid_values /= gridsum
        else:
            self._grid_values /= f

    def write_grid(self, name):
        """
        Write the Grid to a csv file
        """
        outfile = 'grid_' + name + '.csv'
        g = np.vstack((self._grid_points.T, self._grid_values)).T
        np.savetxt(outfile, g, header='x,y,z,d', delimiter=',')


class multigrid(multiprocessing.Process):
    def __init__(self, crd, queue, origin, length, spacing, vdw=False):
        # Init process
        multiprocessing.Process.__init__(self)
        self.vdw = vdw
        # queue:
        self.queue = queue
        # frame coordinates
        self.crd = crd
        # Build Grid
        self.grid = GRID(origin=origin, length=length, spacing=spacing)

    def run(self):
        for frame in self.crd:
            self.grid.map2grid(frame, vdw=self.vdw)
        self.queue.put(self.grid._grid_values)


def get_parser():
    """
    Parses command line using `argparse.ArgumentParser`.

    :param argv: List of command-line arguments without the script name.
    :type argv: `list(str)`

    :return: Namespace with the arguments inside.
    :rtype: `argparse.Namespace`
    """

    run_cmd_str = '$SCHRODINGER/run ' + SCRIPT_FILENAME
    usage = run_cmd_str + ' [options]'
    usage += '\n' + ' ' * (len(usage) - 2) + 'infile outfile'

    p = argparse.ArgumentParser(
        prog=SCRIPT_FILENAME,
        usage=usage,
        description=__doc__)

    p.add_argument('infile',
                   help='Input file name (structure file).',
                   nargs='+')

    p.add_argument('-o',
                   '--outfile',
                   default=None,
                   type=str,
                   metavar='<string>',
                   help='Output file name, can be either .csv, .vis or .ccp4. \n If left black [jobname].vis will be chosen as a filename')

    p.add_argument('-s',
                   '--scale',
                   type=float,
                   default=1.0,
                   metavar='<number>',
                   help='Scale factor for the vdW radii. Default: %(default).2f')

    p.add_argument('-r',
                   '--resolution',
                   type=float,
                   default=0.5,
                   metavar='<number>',
                   help='Volumetric grid resolution. Default: %(default).2fA')

    g = p.add_mutually_exclusive_group()

    g.add_argument(
        '-l',
        '--ligand-asl',
        default='not non_polar_hydrogens',
        metavar='<string>',
        help="ASL expression describing ligand atoms. Default: '%(default)s'.")

    g.add_argument(
        '-a',
        '--auto-ligand',
        action='store_true',
        help='Recognize ligand atoms automatically.'
             ' Cannot be used with -l/--ligand-asl.')

    p.add_argument('-jobname',
                   help='The job name',
                   type=str,
                   metavar='<string>',
                   default=None)

    p.add_argument('-nproc',
                   help='Number of cpus to use',
                   type=int,
                   metavar='<integer>',
                   default=1)

    # Job control options:
    cmdline.add_jobcontrol_options(
        p,
        options=(cmdline.WAIT, cmdline.NOJOBID, cmdline.LOCAL, cmdline.HOST))

    return p


def submit_under_jobcontrol(args, script_args):
    """
    Submit this script under job control.
    """
    # Launch the backend under job control:
    scriptlauncher = launcher.Launcher(
        script=os.path.realpath(__file__),
        jobname=args.jobname,
        local=args.local,
        prog='compute_substrate_envelope',
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


class Backend:
    def __init__(self):
        pass

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)

    def _distribute_work(self, X, N):
        """
        Distribute input array X to N subjobs
        :param X: Number of jobs
        :param N: Number of workers
        :return:
        """
        out = []
        for i in range(N):
            out.append(X[i::N])
        return out

    def _get_vdw_coordinates(self, st):
        """
        Get the atomic coordinates and vdw radius of a atom map set
        Coordinates and radius will be returned in a (N,4) array.
        The first 3 columns are the x,y,z coordinates.
        The final column represents the vdw radius r
        :param st:
        :return:
        """

        # How to identify the ligand
        if self.args.auto_ligand:
            searcher = analyze.AslLigandSearcher()

            def get_ligand_atoms(st):
                for ligand in searcher.search(st):
                    for i in ligand.atom_indexes:
                        yield st.atom[i]
        else:
            def get_ligand_atoms(st):
                for i in analyze.evaluate_asl(st, args.ligand_asl):
                    yield st.atom[i]

        vdw_coord_array = []
        for a in get_ligand_atoms(st):
            vdw_coord_array.append([a.x, a.y, a.z, a.vdw_radius])
        return np.array(vdw_coord_array)

    def run(self, args):
        '''
        The actual production function
        :param args:
        :return:
        '''
        self.info('Command: $SCHRODINGER/run {}'.format(subprocess.list2cmdline(sys.argv)))
        self.args = args
        self.backend = jobcontrol.get_backend()
        if self.backend:
            self.backend.setJobProgress(description='Initializing')

        self.info('Loading input parameters')

        LEGAL_FILES = ['csv', 'vis', 'ccp4']
        outfile = args.outfile
        if outfile.split('.')[-1] not in LEGAL_FILES:
            self.warning('No allowed file format privded saving output as: {}.vis'.format(outfile))
            out_type = 'vis'
        else:
            out_type = outfile.split('.')[-1]

        # Add infile to jlaunch file list:
        infile_total = len(self.args.infile)
        self.info('{} input structures found'.format(infile_total))
        infiles = []
        total_st = 0
        for f in self.args.infile:
            total_st += structure.count_structures(f)
            infiles.append(jobcontrol.get_runtime_path(f))
        # Load structures
        self.info('Loading {} structures from {} input files'.format(total_st, len(infiles)))
        input_coordinates = []
        for f in infiles:
            for st in structure.StructureReader(f):
                input_coordinates.append(self._get_vdw_coordinates(st))
        input_coordinates = np.array(input_coordinates)
        # Determine grid dimensions

        # Buffer zone around the grid
        buffer = 3.0 * np.max([1.0, self.args.scale])
        xmin, xmax = (np.min(input_coordinates.T[0].flatten()) - buffer, np.max(input_coordinates.T[0].flatten()) + buffer)
        ymin, ymax = (np.min(input_coordinates.T[1].flatten()) - buffer, np.max(input_coordinates.T[1].flatten()) + buffer)
        zmin, zmax = (np.min(input_coordinates.T[2].flatten()) - buffer, np.max(input_coordinates.T[2].flatten()) + buffer)
        origin = np.array([xmin, ymin, zmin])
        length = 1 + np.uint((np.array([xmax, ymax, zmax]) - np.array([xmin, ymin, zmin])) / self.args.resolution)
        resolution = [self.args.resolution for _ in range(3)]  # required for voldata

        self.info('Spacing: {}'.format(resolution[0]))
        self.info('Origin: {},{},{}'.format(*origin))
        self.info('Length: {}x{}x{}'.format(*length))

        # Distribute work
        input_coordinates_dist = self._distribute_work(input_coordinates, self.args.nproc)
        # Create an empty grid instance we will save the output fromm all subprocesses here
        grid = GRID(origin=origin, length=length, spacing=resolution)
        self.info('mapping coordinates to grid using {} workers'.format(args.nproc))
        workers = []
        queue = multiprocessing.Queue()
        for i, coord_subset in enumerate(input_coordinates_dist):
            workers.append(multigrid(coord_subset, queue, origin, length, resolution, vdw=True))
            workers[i].start()
            self.info('started worker processing {}'.format(i))
        for n in range(args.nproc):
            grid._grid_values += queue.get()
        for n in range(args.nproc):
            workers[n].join()
        print('final max')
        print(np.max(grid._grid_values))
        # Normalize the grid by the total number of structures used
        grid.normalize(f=total_st)
        print('normal max')
        print(np.max(grid._grid_values))
        # Create voldata object
        dimensions = [len(np.unique(grid._grid_points[::, 0])), len(np.unique(grid._grid_points[::, 1])),
                      len(np.unique(grid._grid_points[::, 2]))]
        grid_points_dict = {}
        for crd, p in zip(grid._grid_points, grid._grid_values):
            grid_points_dict[tuple(np.uint((crd - grid._origin) / grid._spacing))] = p
        grid_points_remapped = np.zeros(dimensions)
        for z in range(dimensions[2]):
            for y in range(dimensions[1]):
                for x in range(dimensions[0]):
                    grid_points_remapped[x][y][z] = grid_points_dict[(x, y, z)]
        voldata = volumedata.VolumeData(N=np.array(dimensions), resolution=resolution, origin=origin)
        voldata.setData(grid_points_remapped.astype('float32'))  # Need float32 for mmsurf
        self.info('Writing output files')
        if out_type == 'vis':
            volumedataio.SaveVisFile(voldata, outfile)
        elif out_type == 'ccp4':
            volumedataio.SaveCCP4File(voldata, outfile)
        else:
            np.savetxt(outfile, np.hstack((grid._grid_points, grid._grid_values)), header='x,y,z,d', delimiter=',')
        self.backend.addOutputFile(outfile)
        self.info('Output files:' + outfile)


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

    # Do some error checking
    # If no jobname was provided name after outfile
    if args.jobname is None:
        args.jobname = fileutils.get_jobname(args.outfile)

    if not args.scale > 0.0:
        parser.error('argument -s/--scale: must be positive')

    if not args.resolution > 0.0:
        parser.error('argument -r/--resolution: must be positive')

    if not args.auto_ligand:
        if not analyze.validate_asl(args.ligand_asl):
            parser.error('argument -l/--ligand-asl: not a valid ASL')

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
        app = Backend()
        app.run(args)
