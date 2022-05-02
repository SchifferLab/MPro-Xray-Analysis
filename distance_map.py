from __future__ import print_function, division

import sys

import numpy as np
import pandas as pd

import argparse
import warnings

try:
    from schrodinger import structure
    from schrodinger.structutils import analyze
except:
    warnings.warn('Could not find schrodinger modules.\nReading structures not possible.')

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from matplotlib import rc


# Set mpl fonts
matplotlib.rcParams.update({'font.size': 18})
# Increase spacing between ticks
matplotlib.rcParams.update({'xtick.major.pad': 10})
matplotlib.rcParams.update({'ytick.major.pad': 4})

desc = '''
Calculate the difference between two distance plots
'''

def get_parser():
    ap = argparse.ArgumentParser(prog='distance_difference',description=desc)
    ap.add_argument('infiles',
                    type=str,
                    nargs='+',
                    help='Input file can either be structure file or table')
    ap.add_argument('--asl',
                    type=str,
                    nargs ='+',
                    default=['a. CA', ],
                    help='Subset of atoms to use in maestro atom selection language\n Multiple arguments can be passed')
    ap.add_argument('-o',
                    '--outname',
                    default='distance_difference',
                    help='Prefix added to the outfiles')
    ap.add_argument('--heatmap',
                    action='store_true',
                    default=False,
                    help='When provided the distance difference map will be ploted')
    ap.add_argument('--write_distmap',
                    action='store_true',
                    default=False,
                    help='When provided the distance maps will be saved')
    ap.add_argument('--no_difference_map',
                    action='store_true',
                    default=False,
                    help='When provided no distance difference maps will be generated')
    return ap



def plot_heatmap(data, outname=None, title='', xlabel='', ylabel='', column_labels=None, row_labels=None,
                 hide_middleline=True, color_scheme='jet', cutoff=0, vmin=None, vmax=None, nan_color='gray' ):
    """

    :param data:
    :param outname:
    :param title:
    :param xlabel:
    :param ylabel:
    :param column_labels:
    :param row_labels:
    :param hide_middleline:
    :param color_scheme:
    :param cutoff:
    :param vmin:
    :param vmax:
    :param nan_color:
    :return:
    """
    ##                            ##
    ##some initial data processing##
    ##                            ##
    if column_labels is None:
        column_labels = list(10 * x for x in range(len(data[0]) // 10))
        column_labels.append(len(data[0]))
    else:
        column_labels = column_labels.values
        cl_is_unnamed = np.abs(~np.core.defchararray.find(column_labels.astype(str),'Unnamed')).astype(bool)
        column_labels[np.logical_or(pd.isnull(column_labels),cl_is_unnamed)] = ''

    if row_labels is None:
        row_labels = list(10 * x for x in range(len(data) // 10))
        row_labels.append(len(data))
    else:
        row_labels = row_labels.values
        rl_is_unnamed = np.abs(~np.core.defchararray.find(row_labels.astype(str), 'Unnamed')).astype(bool)
        row_labels[np.logical_or(pd.isnull(row_labels), rl_is_unnamed)] = ''
    #
    # If hidde_middleline == True we mask the middle line
    mask = np.zeros(data.shape, dtype=bool)
    # mask all values below cutoff
    mask[np.logical_or((np.abs(data) < cutoff),(np.isnan(data)))] = 1
    if hide_middleline:
        for x in range(len(data)):
            mask[x][x] = 1
    data_masked = np.ma.array(data, mask=mask)
    ##                ##
    ##"build" the plot##
    ##                ##
    #
    # Set color scheme
    cmap = plt.get_cmap(color_scheme)
    cmap.set_bad(color='w', alpha=0.)
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))
    #
    if vmin is None:
        vmin=-np.abs(data_masked).max()
        vmax=np.abs(data_masked).max()
    #
    heatmap = ax.pcolormesh(data_masked, cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
    if nan_color is not None:
        nan_cells = np.ones(data.shape)
        nan_cells_masked = np.ma.array(nan_cells, mask=~np.isnan(data))
        nans = ax.pcolormesh(nan_cells_masked,cmap=nan_color,zorder=10, alpha=0.7)
    #
    lim1 = 0.0
    lim2 = len(data_masked)
    ax.set_xlim((lim1, lim2))
    ax.set_ylim((lim1, lim2))
    ax.axis('equal')
    #
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    plt.colorbar(heatmap, cax=cax)
    #ax.plot([x for x in range(len(data_masked))], [y for y in range(len(data_masked))], linewidth=2, c='k')
    #
    # Title & Labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #
    ##
    ##adjust properites
    ##
    #
    # turn off the frame
    ax.set_frame_on(False)
    #
    # show squares => use equal spacing for x and y axis
    #
    # Don't show x,y axis grid
    ax.grid(False)
    #
    ax.set_xticks(np.arange(len(column_labels)), minor=False)
    ax.set_xticklabels(column_labels, rotation=90)
    ax.set_yticks(np.arange(len(row_labels)), minor=False)
    ax.set_yticklabels(row_labels)
    #
    # some aestethic adjustments
    ax.invert_yaxis()
    ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()
    #
    if outname is None:
        plt.show()
        plt.close()
    else:
        pngName = outname + '.png'
        plt.savefig(pngName, dpi=300)
        plt.close()


def calc_ca_dist(fn, asl='a.pt CA'):
    """

    :param fn:
    :param asl:
    :return:
    """
    # measure distance
    def _measure_distance(crd1, crd2):
        return np.sqrt((np.sum((crd1 - crd2) ** 2)))

    st = structure.Structure.read(fn)
    atoms = [a for a in st.atom if a.index in analyze.evaluate_asl(st, asl)]
    natoms = len(atoms)
    # create output array
    distm = np.zeros((natoms, natoms))
    atom_ids = []
    for i in range(natoms):
        a1 = atoms[i]
        atom_ids.append((a1.pdbname.strip(), a1.resnum, a1.chain.strip()))
        for j in  range(i,natoms):
            a2 = atoms[j]
            r = _measure_distance(np.array(a1.xyz), np.array(a2.xyz))
            distm[i][j] = r
            distm[j][i] = r
    return atom_ids, distm


def main(args):
    # Input files
    if len(args.infiles) == 1:
        if not args.heatmap:
            warnings.warn('Provide more than one infile or --heatmap')
            sys.exit(0)
        ddmap = pd.read_table(args.infiles[0], index_col=0, sep=',')
    else:
        nfiles = len(args.infiles)
        cmaps = []
        if len(args.asl) < nfiles and len(args.asl) != 1:
            warnings.warn('More than one asl passed but not as many as atructures', category=RuntimeWarning)
            asl = args.asl
            for i in range(nfiles-len(args.asl)):
                asl.append(asl[0])
        elif len(args.asl) == 1 and nfiles != 1:
            asl = [args.asl[0] for _ in range(nfiles)]
        else:
            asl = args.asl

        for i, fn in enumerate(args.infiles):
            cmaps.append(calc_ca_dist(fn,asl[i]))


        for i in range(nfiles-1):
            fn1 = args.infiles[i].split('.')[0]
            index1,dmap1 = cmaps[i]
            columns = ['{}:{}:{}'.format(*aid) for aid in index1]
            df1 = pd.DataFrame(dmap1, index=columns, columns=columns)
            if args.write_distmap:
                csv1 = '{}_distance_map.csv'.format(fn1)
                df1.to_csv(csv1,sep=',')
            for j in range(i+1,nfiles):
                if args.no_difference_map:
                    continue
                index2,dmap2 = cmaps[j]
                columns = ['{}:{}:{}'.format(*aid) for aid in index2]
                df2 = pd.DataFrame(dmap2, index=columns, columns=columns)
                fn2 = args.infiles[j].split('.')[0]
                #  Get all atoms common accros distance maps 1 & 2
                is_common = df1.columns.isin(df2.columns)
                #  Get common atom_ids
                ids_common = df1.columns[is_common]
                #  Total count common elements
                ncommon = np.count_nonzero(is_common)
                ddmap = pd.DataFrame(np.zeros((ncommon,ncommon)), columns=ids_common, index=ids_common)
                for aid1 in ids_common:
                    for aid2 in ids_common:
                        ddmap[aid1][aid2] = df1[aid1][aid2] - df2[aid1][aid2]
                csv_outname = '{}_{}_distance_difference.csv'.format(fn1, fn2)
                ddmap.to_csv(csv_outname, sep=',')
    if args.heatmap:
        plot_heatmap(ddmap.values, args.outname,
                     column_labels=ddmap.columns,
                     row_labels=ddmap.index)

if __name__ == '__main__':
    ap = get_parser()
    args = ap.parse_args()
    main(args)
