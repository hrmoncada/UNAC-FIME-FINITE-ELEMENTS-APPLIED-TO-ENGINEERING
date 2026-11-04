#!/usr/bin/env python3
"""
contour_sequence4.py

Python translation of John Burkardt's MATLAB program
CONTOUR_SEQUENCE4.

This program

    * Reads an XY grid file (XY coordinate file).
          xy.txt
    * Reads a sequence of scalar field files
          u00001.txt
          u00002.txt
          ...
    * Computes the global minimum and maximum.
    * Produces filled contour plots.
    * Saves every frame as a PNG image.

Author:
    MATLAB version: John Burkardt
    Python translation: ChatGPT

it can be executed as

python contour_sequence4.py xy.txt u00001.txt

Read all the files with following structure: 
 u00001.txt  ...  u00005.txt  ... u00010.txt

Create an output file 
   contour_sequence4_test.txt
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utility routines
# ------------------------------------------------------------

def file_exist(filename):
    """
    Return True if the file exists.
    """
    return Path(filename).is_file()


def ch_is_digit(c):
    """
    Return True if character is a decimal digit.
    """
    return c.isdigit()


def digit_inc(c):
    """
    Increment one digit.

    0->1
    ...
    8->9
    9->0
    """
    if c == '9':
        return '0'
    elif '0' <= c <= '8':
        return str(int(c) + 1)
    else:
        return c


def s_len_trim(s):
    """
    Length of string up to last nonblank.
    """
    return len(s.rstrip())


def timestamp():
    """
    Print current date/time.
    """
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------
# Filename utilities
# ------------------------------------------------------------

def file_name_ext_swap(filename, new_ext):
    """
    Replace extension of filename.

    Example
    -------
    data.txt -> data.png
    """

    p = Path(filename)

    if not new_ext.startswith("."):
        new_ext = "." + new_ext

    return str(p.with_suffix(new_ext))


def file_name_inc(filename):
    """
    Increment every digit string exactly as the MATLAB routine.

    Examples
    --------

    u00001.txt -> u00002.txt
    u00009.txt -> u00010.txt
    u00999.txt -> u01000.txt
    u99999.txt -> u00000.txt

    If there are no digits, returns ''.
    """

    chars = list(filename)

    found_digit = False
    carry = True

    for i in range(len(chars) - 1, -1, -1):

        if chars[i].isdigit():

            found_digit = True

            if carry:

                if chars[i] == '9':
                    chars[i] = '0'
                    carry = True
                else:
                    chars[i] = str(int(chars[i]) + 1)
                    carry = False

    if not found_digit:
        return ""

    return "".join(chars)

# ------------------------------------------------------------
# Determine mesh orientation
# ------------------------------------------------------------

def detect_grid(xy):
    """
    Determine NX and NY from the XY file.

    Parameters
    ----------
    xy : ndarray (m,2)

    Returns
    -------
    orientation
        1 -> X constant first
        2 -> Y constant first

    nx
    ny
    """

    m = xy.shape[0]

    #
    # MATLAB:
    #
    # if xy(2,1)==xy(1,1)
    #

    if np.isclose(xy[1, 0], xy[0, 0]):

        orientation = 1

        ny = 2

        while ny < m:

            if not np.isclose(xy[ny, 0], xy[0, 0]):
                break

            ny += 1

        nx = m // ny

    else:

        orientation = 2

        nx = 2

        while nx < m:

            if not np.isclose(xy[nx, 1], xy[0, 1]):
                break

            nx += 1

        ny = m // nx

    return orientation, nx, ny


# ------------------------------------------------------------
# Read all scalar files to determine global limits
# ------------------------------------------------------------

def compute_global_limits(first_u_file):
    """
    Scan every U file to determine the global
    minimum and maximum values.
    """

    filename = first_u_file

    u = np.loadtxt(filename)

    umin = np.min(u)
    umax = np.max(u)

    while True:

        filename = file_name_inc(filename)

        if filename == "":
            break

        if not file_exist(filename):
            break

        u = np.loadtxt(filename)

        umin = min(umin, np.min(u))
        umax = max(umax, np.max(u))

    return umin, umax

# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------

def contour_sequence4(xy_filename="xy.txt", u_filename="u00001.txt"):
    """
    Create contour plots from a sequence of scalar data files.

    Parameters
    ----------
    xy_filename : str
        File containing X,Y coordinates.

    u_filename : str
        First scalar field file.
    """

    print()
    print("CONTOUR_SEQUENCE4")
    print("  Python version")
    print("  Filled contour plots from a sequence of datasets.")
    print()

    #
    # Check input files.
    #
    if not file_exist(xy_filename):
        raise FileNotFoundError(
            f'XY file "{xy_filename}" does not exist.')

    if not file_exist(u_filename):
        raise FileNotFoundError(
            f'U file "{u_filename}" does not exist.')

    #
    # Save original names.
    #
    u_root = u_filename
    png_filename = file_name_ext_swap(u_root, "png")

    print(f"First PNG file : {png_filename}")

    #
    # Read XY coordinates.
    #
    xy = np.loadtxt(xy_filename)

    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            "XY file must contain at least two columns.")

    #
    # Read first scalar field.
    #
    u = np.loadtxt(u_filename)

    if u.ndim > 1:
        u = u[:, 0]

    m = len(u)

    #
    # Determine mesh dimensions.
    #
    orientation, nx, ny = detect_grid(xy)

    print()
    print(f"Orientation : {orientation}")
    print(f"NX          : {nx}")
    print(f"NY          : {ny}")

    #
    # Determine global contour limits.
    #
    print()
    print("Scanning scalar files...")

    umin, umax = compute_global_limits(u_root)

    print(f"Global minimum : {umin:e}")
    print(f"Global maximum : {umax:e}")

    #
    # Number of contour intervals.
    #
    nlevels = 15

    levels = np.linspace( umin, umax, nlevels + 1 )

    #
    # Restart from first scalar field.
    #
    u_filename = u_root

    frame_num = 0

    #
    # Plot every scalar field.
    #
    while True:
        if not file_exist(u_filename):
            break

        print(f"Processing {u_filename}")

        u = np.loadtxt(u_filename)

        if u.ndim > 1:
            u = u[:, 0]

        #
        # Reshape arrays exactly as MATLAB.
        #
        if orientation == 1:
            x = xy[:, 0].reshape((nx, ny))
            y = xy[:, 1].reshape((nx, ny))
            z = u.reshape((nx, ny))
        else:
            x = xy[:, 0].reshape((ny, nx)).T
            y = xy[:, 1].reshape((ny, nx)).T
            z = u.reshape((ny, nx)).T

        #
        # PLOT Create figure.
        #
        fig, ax = plt.subplots( figsize=(8, 6), dpi=150 )
        contour = ax.contourf( x, y, z,levels=levels, cmap="jet", vmin=umin, vmax=umax )
        # Add grid properly on the axes
        ax.grid(True)
        # Color bar.
        cbar = fig.colorbar(contour)
        cbar.set_label("U")
        # Labels.
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title( Path(u_filename).name,  fontsize=14, fontweight="bold"  )
        ax.set_aspect("equal")
        fig.tight_layout()

        # Save figure as PNG.
        fig.savefig( png_filename, dpi=150, bbox_inches="tight" )
        print(f"  Saved {png_filename}")

        plt.close(fig)

        frame_num += 1

        #
        # Prepare next filenames.
        #
        next_u = file_name_inc(u_filename)

        #
        # Stop if no more files exist.
        #
        if next_u == "":
            break

        if not file_exist(next_u):
            break

        u_filename = next_u
        png_filename = file_name_ext_swap(
            u_filename,
            "png"
        )

    #
    # Finished.
    #
    print()
    print(f"Processed {frame_num} frame(s).")
    print()
    print("CONTOUR_SEQUENCE4")
    print("Normal end of execution.")


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():

    #
    # Default filenames.
    #
    xy_filename = "xy.txt"
    u_filename = "u00001.txt"

    #
    # Command-line arguments.
    #
    if len(sys.argv) >= 2:
        xy_filename = sys.argv[1]

    if len(sys.argv) >= 3:
        u_filename = sys.argv[2]

    contour_sequence4(
        xy_filename,
        u_filename
    )


if __name__ == "__main__":

    timestamp()
    main()
    timestamp()
# ------------------------------------------------------------
# Replacement for file_name_inc()
# ------------------------------------------------------------

def file_name_inc(filename):
    """
    Increment the numeric portion of a filename exactly as John Burkardt's MATLAB FILE_NAME_INC.

    Examples
    --------
    u00001.txt -> u00002.txt
    u00009.txt -> u00010.txt
    u00999.txt -> u01000.txt
    u99999.txt -> u00000.txt

    If the filename contains no digits, return "".
    """

    if len(filename) == 0:
        raise ValueError("Input filename is empty.")

    chars = list(filename)

    change = 0

    #
    # Scan from right to left.
    #
    for i in range(len(chars) - 1, -1, -1):

        c = chars[i]

        if '0' <= c <= '8':

            change += 1
            chars[i] = chr(ord(c) + 1)

            return "".join(chars)

        elif c == '9':

            change += 1
            chars[i] = '0'

    #
    # No digits at all.
    #
    if change == 0:
        return ""

    #
    # All digits wrapped around.
    #
    return "".join(chars)
