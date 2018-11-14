#!/usr/bin/env python

from __future__ import print_function, division

import os.path
import sys
import argparse
import collections
import re
import math
import numpy as N

def gehrels(c):
    return 1. + N.sqrt(c + 0.75)

SummedProfile = collections.namedtuple(
    'SummedProfile', ['radii', 'widths', 'cts', 'areaexp','exposure'])

def readSumProfiles(infilenames):
    """Read in profiles and sum.

    Returns radii of bins, half widths, total counts and area*exposure
    """

    data = N.loadtxt(infilenames)
    radii = data[:,0]
    widths = data[:,1]
    exposure = data[:,4]
    totcts = data[:,2]
    totareaexp = data[:,3]*data[:,4]

    assert N.allclose(radii, data[:,0])
    assert N.allclose(widths, data[:,1])

    totcts += data[:,2]
    totareaexp += data[:,3]*data[:,4]

    return SummedProfile(radii, widths, totcts, totareaexp, exposure)

def doBinning(fgprofs, bgprofs, snthresh):
    """Bin up to signal noise given and splits its in nbins

    Returns outer bin number for each bin
    """

    nbins = len(fgprofs.radii)
    outradii = []
    lastidx = 0
    ratio = N.sum(fgprofs.exposure)/(N.sum(bgprofs.exposure))

    for i in range(1, nbins+1):
        totfgcts = fgprofs.cts[lastidx:i].sum()
        totfgareaexp = fgprofs.areaexp[lastidx:i].sum()
        totbgcts = bgprofs.cts[lastidx:i].sum()
        totbgareaexp = bgprofs.areaexp[lastidx:i].sum()

        signal = totfgcts/totfgareaexp - totbgcts/totbgareaexp
        noise = math.sqrt(
            (gehrels(totfgcts)/totfgareaexp)**2 +
            (gehrels(totbgcts)/totbgareaexp)**2)

        sn = signal / noise
        if sn >= snthresh or i == nbins:
            outradii.append(i)
            lastidx = i

    print('Produced', len(outradii), 'sn:', snthresh)
    return outradii


def writeBinnedProfile(infilename, outfilename, radii, header=''):
    """Given a list of input bins (giving the outer indices), bin
    input profile."""

    indata = N.loadtxt(infilename)
    outdata = []
    lastidx = 0
    for i in radii:
        inrad = indata[lastidx,0]-indata[lastidx,1]
        outrad = indata[i-1,0]+indata[i-1,1]
        cts = indata[lastidx:i, 2]
        area = indata[lastidx:i, 3]
        exp = indata[lastidx:i, 4]


        avexp = (area*exp).sum() / area.sum()

        outdata.append(
            [ 0.5*(inrad+outrad),
              0.5*(outrad-inrad),
              cts.sum(),
              area.sum(),
              avexp,
              cts.sum() / avexp / area.sum(),
              ]
            )
        lastidx = i

    N.savetxt(outfilename, outdata, fmt='%e', header=header)

def writeAllBinnedProfiles(radii,fgprofs,bgprofs,sn,name,sufix):
    """Write all binned profiles."""

    # v = vars(args)
    # header = ['Binned output using %s\n' % os.path.basename(sys.argv[0])]
    header = ['Binned output\n']
    #for k in sorted(v):
    #    header.append(' %s = %s\n' % (k, v[k]))

    #filenames = args.inprofile + args.backprof

    # outfilename = filename + args.suffix
    # oufilename = inprofile+sufix
    for i in range(2):
        outfilename = name[i].split('.dat')[0]+sufix
        # print('Binning', filename, 'to', outfilename)

        hdr = header + [
            'input file: %s\n' % name[i],
            'output file: %s\n' % outfilename,
            '\n',
            'rcentre(amin) rhalfwidth(amin) counts area(amin2) exposure sb\n']

        writeBinnedProfile(name[i], outfilename, radii, header=''.join(hdr))

class AddSplitAction(argparse.Action):
    """Add to a list of items, splitting on whitespace or commas."""
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, [])
        ls = getattr(namespace, self.dest)
        ls += [a for a in re.split(r'[, \n\t]', values) if a]

def SNbinning(inprofile,backprof,sn,sufix):

    # read inputs
    fgprofs = readSumProfiles(inprofile)

    # optionally read a background inputs
    bgprofs = readSumProfiles(backprof)
    assert(N.allclose(bgprofs.radii, fgprofs.radii))
    assert(N.allclose(bgprofs.widths, fgprofs.widths))

    # do the binning
    radii = doBinning(fgprofs, bgprofs, sn)
    name = inprofile, backprof
    writeAllBinnedProfiles(radii,fgprofs,bgprofs,sn,name,sufix)

# # if __name__ == '__main__':
#     main()
