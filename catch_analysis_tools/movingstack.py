from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
import statistics
import json
import os
import numpy as np
import warnings
import sep

import argparse
import astropy
import astropy.units as u

defaultzeropoint = 25

def create_wcs(ra, dec, size_x, size_y, pixel_scale=0.4):
    """
    Create an undistorted WCS object based on a given reference position and image size.
    
    Parameters:
    ra : float
        Right Ascension (RA) of the reference position in degrees.

    dec : float
        Declination (Dec) of the reference position in degrees.
    size_x : int 
        Number of pixels along the x-axis (image width).
    size_y : int 
        Number of pixels along the y-axis (image height).
    pixel_scale :  float 
        Pixel scale in arcseconds per pixel (default is 0.4"/pixel).
    
    Returns:
    wcs : astropy.wcs.WCS
        An astropy WCS object.
    """

    # Create a WCS object
    wcs = WCS(naxis=2)

    # Set the reference pixel to the center of the image
    wcs.wcs.crpix = [size_x / 2, size_y / 2]

    # Set the reference position (CRVAL) in degrees
    wcs.wcs.crval = [ra, dec]

    # Set the pixel scale (CDELT) in degrees per pixel
    #wcs.wcs.cdelt = np.array([-pixel_scale / 3600, pixel_scale / 3600])
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])

    # Set the coordinate system to RA/Dec with degrees as units
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return wcs

def project_fits_to_wcs(input_fits, new_wcs, shape_out, photometric_cal=True, subtract=True, intermediate=False):
    """
    Project a FITS image onto a new WCS using the reproject library.

    Parameters:
    input_fits : str
        Path to the input FITS file.
    new_wcs : astropy.wcs.WCS
        The target WCS object.
    shape_out :tuple 
        The shape (height, width) of the output image.

    Returns:
    reprojected_data : numpy.array(ndim=2)
        The image reprojected onto the new WCS.
    """
    # Open the input FITS file and extract the data and WCS
    with fits.open(input_fits) as hdu_list:
      input_data = hdu_list[0].data
      header = hdu_list[0].header
    original_wcs = get_wcs(input_fits)

    # Photometric calibration
    if (photometric_cal):
      magzp = get_photozp(header)
      input_data = input_data * 10**(-0.4 * (magzp - defaultzeropoint))
      if (intermediate):
        hdu=fits.PrimaryHDU(data=input_data,header=header)
        hdu.writeto(os.getcwd()+"/"+os.path.splitext(os.path.basename(input_fits))[0]+".photocal.fit",overwrite=True)

    # Subract background
    if (subtract):
      input_data = input_data.byteswap().newbyteorder()
      try:
        bkg = sep.Background(input_data)
      except ValueError:
        input_data = input_data.byteswap().newbyteorder()
        bkg = sep.Background(input_data)
      input_data = input_data - bkg
      if (intermediate):
        hdu=fits.PrimaryHDU(data=bkg,header=header)
        hdu.writeto(os.getcwd()+"/"+os.path.splitext(os.path.basename(input_fits))[0]+".bkg.fit",overwrite=True)
        hdu=fits.PrimaryHDU(data=input_data,header=header)
        hdu.writeto(os.getcwd()+"/"+os.path.splitext(os.path.basename(input_fits))[0]+".subtracted.fit",overwrite=True)

    # Reproject the image onto the new WCS
    reprojected_data, _ = reproject_interp((input_data, original_wcs), new_wcs, shape_out=shape_out)

    return reprojected_data

def get_photozp(header):
  """
  Get a photometric zero point from a fits header (return default if not found)

  Parameters:
  header : astropy.io.fits.Header
    A fits header object

  Returns:
  zeropoint : float
    photometric zero point
  """
  try:
    return header["MAGZEROP"]
  except:
    pass
  try:
    return header["MAGZP"]
  except:
    pass
  try:
    return header["ZPAPPROX"]
  except:
    pass
  try:
    return header["MAGZPT"]
  except:
    return defaultzeropoint

def get_wcs(input_fits):
  """
  Get a World Coordinate System (WCS) object from a fits file

  Parameters:
  input_fits : file
    a fits file

  Returns:
  wcs : astropy.wcs.WCS
    An astropy WCS object.
  """
  with fits.open(input_fits) as hdu_list:
    try:
      return WCS(hdu_list[0].header)
    except ValueError:
      wcs = WCS(naxis=2)
      h = hdu_list[0].header
      wcs.wcs.ctype = h["CTYPE1"], h["CTYPE2"]
      wcs.wcs.crval = h["CRVAL1"], h["CRVAL2"]
      wcs.wcs.crpix = h["CRPIX1"], h["CRPIX2"]
      wcs.wcs.cdelt = h["CDELT1"], h["CDELT2"]
      wcs.array_shape = h["NAXIS1"], h["NAXIS2"]
      return wcs

def get_minpixsc(wcs):
  """
  Get the minimum pixel scale of the image

  Parameters:
  wcs : astropy.wcs.WCS
    An astropy WCS object.

  Returns:
  pixelscale : float
    The minimum pixel scale
  """
  try:
    wcs.wcs.cd
    return np.sort(np.abs(wcs.wcs.cd.flatten()))[-2]
  except AttributeError:
    return min(wcs.wcs.cdelt)
    
def get_maxpixsc(wcs):
  """
  Get the maximum pixel scale of the image

  Parameters:
  wcs : astropy.wcs.WCS
    An astropy WCS object.

  Returns:
  pixelscale : float
    The maximum pixel scale
  """
  try:
    wcs.wcs.cd
    return np.sort(np.abs(wcs.wcs.cd.flatten()))[-1]
  except AttributeError:
    return max(wcs.wcs.cdelt)


def unquoted_string(arg):
    """
    Helper function to remove quotes from a string

    Parameters:
    arg : string

    Returns:
    stripped : string
    """
    return arg.strip('\'"')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',required=True,type=unquoted_string, help="the path to the CATCH json file for the images to be stacked")
parser.add_argument('--json',required=True,type=unquoted_string,help="the json file generated by CATCH")
parser.add_argument('--basename',required=True,type=unquoted_string,help="the the base to use for output filenames")
args = parser.parse_args()
dataroot = args.dataroot
jsonname = args.json
basename = args.basename

#get base name and check for uniformity
with open(dataroot+jsonname, 'r') as f:
  jdata = json.load(f)


###  Get the maximum size and minimum pixel scale required for a WCS to capture input images footprints with minimal data loss
wcs = [get_wcs(dataroot+"fits/"+frm["product_id"]+".fit") for frm in jdata]
#for w in wcs:
#  print(w.wcs.cd)
#  print(np.abs(w.wcs.cd.flatten()))
#  print(np.sort(np.abs(w.wcs.cd.flatten()))[-2])
minpixsc = min([get_minpixsc(w) for w in wcs])
maxpixsc = max([get_maxpixsc(w) for w in wcs])
maxsz = round(max([max(w.array_shape) for w in wcs])*maxpixsc/minpixsc*2**.5)
#print(maxsz)

print("Projecting each image onto undistorted WCS")
projected = []
for frm in jdata:
  targetwcs = create_wcs(frm["ra"],frm["dec"],maxsz,maxsz,minpixsc)
  #hdu0=fits.open(dataroot+"fits/"+frm["product_id"]+".fit")
  print("Projecting "+dataroot+"fits/"+frm["product_id"]+".fit")

  #print("checking coords: target wcs pix coords at detection")
  #print(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(frm["ra"]*u.degree,frm["dec"]*u.degree),targetwcs))
  #print("checking coords: origin wcs pix coords at detection")
  #with fits.open(dataroot+"fits/"+frm["product_id"]+".fit") as hdu1:
  #  originwcs = WCS(hdu1[0])
  #  print(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(frm["ra"]*u.degree,frm["dec"]*u.degree),originwcs))
  #  projectedTautology = project_fits_to_wcs(dataroot+"fits/"+frm["product_id"]+".fit",originwcs,(maxsz,maxsz))

  projectedlocal = project_fits_to_wcs(dataroot+"fits/"+frm["product_id"]+".fit",targetwcs,(maxsz,maxsz))
  projected.append(projectedlocal)
  
  #hdu=fits.PrimaryHDU(data=projectedlocal,header=targetwcs.to_header())
  #hdu.writeto(os.getcwd()+"/"+frm["product_id"]+"_projected.fit",overwrite=True)
  #hduT=fits.PrimaryHDU(data=projectedTautology,header=originwcs.to_header())
  #hduT.writeto(os.getcwd()+"/"+frm["product_id"]+"_tautology.fit",overwrite=True)
  #print("wcsdata")
  #print(originwcs)
  #print("targwcs")
  #print(targetwcs)


###do mean for each pixel and return
ra = sum([frm['ra'] for frm in jdata])/len(jdata)
dec= sum([frm['dec'] for frm in jdata])/len(jdata)


### assemble the per pixel median image from the reprojected inputs
finalwcs = create_wcs(ra,dec,maxsz,maxsz,minpixsc)
header = finalwcs.to_header()
imagedata = np.zeros(shape=(maxsz,maxsz))
for i in range(imagedata.shape[0]):
  for j in range(imagedata.shape[1]):
    #print(projected[0][i,j])
    imagedata[i,j] = statistics.median([proj[i,j] for proj in projected])

hdu=fits.PrimaryHDU(data=imagedata,header=header)
hdu.writeto(os.getcwd()+"/"+basename+"_stack.fit",overwrite=True)


