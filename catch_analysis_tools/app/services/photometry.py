import os
from catch_analysis_tools.photometry import get_image,subpixel_centroid,create_user_aperture,define_aperture,do_aperture_photometry,source_instr_mag,calibrated_mag
from catch_analysis_tools.background import calc_bkg
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.io import fits
from catch_analysis_tools.background import global_subtraction

def get_world_coordinates(WCS_file,x,y):
    """
    Accepts an astropy-readable WCS object (a .wcs or fits file ) and outputs the world coordinates of an (x,y) pixel point in decimal degrees.
    """
    #WCS_file = body['WCS_file']
    #x = body['x']
    #y = body['y']
    world_coords = WCS(fits.open(WCS_file)[0].header)
    loc = world_coords.pixel_to_world(x,y)
    print(loc)
    transform_results = {
        "x":x,
        "y":y,
        "ra":loc.ra.deg,
        "dec":loc.dec.deg
    }
    return transform_results

def get_pixel_coordinates(WCS_file,ra,dec):
    """
    Accepts an astropy-readable WCS object (a .wcs or fits file) and outputs the (x,y) pixel coordinates of an (ra,dec) point in decimal degrees.
    """
    #WCS_file = body['WCS_file']
    #ra = body['ra']
    #dec = body['dec']
    world_coords = WCS(fits.open(WCS_file)[0].header)
    sky_loc = SkyCoord(ICRS(ra=ra*u.deg, dec=dec*u.deg))
    loc = world_coords.world_to_pixel(sky_loc)
    x = loc[0].item()
    y = loc[1].item()

    transform_results = {
        "x":x,
        "y":y,
        "ra":ra,
        "dec":dec
    }
    return transform_results  

def centroid(file,target_x,target_y,search_radius):
    """
    Searches for source nearby to expected ephemeris location in image, assumes that astrometry solution has been rerun and that both the cutout .fits file and the redone .wcs file exist. 
    
    
    Parameters
    ----------
    file : string
        Base filename (without .fits or .wcs extension) taken from CATCH-generated query URL.

    target_x : float
        Target x pixel location, to be used as initial guess for the object.

    target_y : float
        Target y pixel ephemeris location, to be used as initial guess for the object.

    search_radius : float
        Radius in pixels of search area (centered on (x,y) ) for centroiding


    Returns
    -------
    search_results :
        array_like
                    
                    
    figname: 
        string
                    Output plot showing the default aperture + annulus extraction onto the cutout image.
    """

    
    #tmp_wcs = WCS(file+'.wcs')
    img, header = get_image(file)

    #target_pix = get_WCS_pixel(tmp_wcs,target_ra,target_dec)
    #target_x = target_pix[0].item()
    #target_y = target_pix[1].item()

    cent_pix = subpixel_centroid([target_x,target_y],img,search_radius)
    #cent_loc = get_pixel_WCS(tmp_wcs,cent_pix[0],cent_pix[1])
    
    search_results = {
        "init_guess_x":target_x,
        "init_guess_y":target_y,
        "search_radius":search_radius,
        "cent_x":cent_pix[0],
        "cent_y":cent_pix[1],
    }
    print(search_results)
    interval = ZScaleInterval()
    norm = ImageNormalize(img, interval=ZScaleInterval())

            
    plt.figure(figsize=(8,8))
    plt.imshow(img,norm=norm,cmap='gray_r')
    plt.scatter(img.shape[0]/2,img.shape[1]/2,s=100,marker='x',label='Image Center')
    plt.scatter(target_x,target_y,s=100,marker='+',label='Initial Guess')
    plt.scatter(cent_pix[0],cent_pix[1],s=50,marker='.',c='yellow',label='Centroid Location')
    plt.xlim(target_x-20,target_x+20)
    plt.ylim(target_y-20,target_y+20)
    plt.legend()

        

    file_base = os.path.splitext(file)[0]
    figname = 'centroid.png'
    plt.savefig(figname)
    plt.close()
    centroid_results = {
        "search_results": search_results,
        "centroid_figure": figname
    }

    return centroid_results

#todo: create function that takes target location, background location and outputs aperture objects
# this function can be fed the outputs of centroid_location

def target_extraction(body):
    """
    Searches for source nearby to expected ephemeris location in image, assumes that astrometry solution has been rerun and that both the cutout .fits file and the redone .wcs file exist. 
    
    
    Parameters
    ----------
    body : dict
        Request body containing file, target_aperture_params, and background_aperture_params.
    """
    file = body['file']
    target_aperture_params = body['target_aperture_params']
    background_aperture_params = body['background_aperture_params']


    #tmp_wcs = WCS(filebase+'.wcs')
    img, header = get_image(file)


    data_sub, twoD_bkg = global_subtraction(img)
    file_base = os.path.splitext(file)[0]
    #targ_loc = get_WCS_pixel(tmp_wcs,target_ra,target_dec)

    #centroid_results = centroid_location(filebase,target_x,target_y,search_radius)
    #targ_cent = np.array([centroid_results["search_results"]["cent_x"],centroid_results["search_results"]["cent_y"]])

    target_aperture = define_aperture(target_aperture_params)
    background_aperture = define_aperture(background_aperture_params)

    interval = ZScaleInterval()
    norm = ImageNormalize(data_sub, interval=ZScaleInterval())

    bkg, bkg_var = calc_bkg(data_sub,background_aperture,'median',None)
            
    target_flux, target_fluxerr = do_aperture_photometry(img,target_aperture,background_aperture)
    #instr_mag = source_instr_mag(aperture_flux,aperture_fluxerr,1)
    #cal_mag = calibrated_mag(instr_mag,header['zp'],header['zp_std']) # temporarily using the original Atlas header values

    #cal_mag_array = calibrated_mag(instr_mag,header['magzpt'],header['zprms'])
    
    # could put code to filter out frames where targ_loc and targ_cent vary by more than a couple pixels here
    # (would mean star hit)
            
    plt.figure(figsize=(8,8))
    plt.imshow(data_sub,norm=norm,cmap='gray_r')
    target_aperture.plot(color='blue', lw=1.5, alpha=0.5)
    background_aperture.plot(color='yellow',lw=1.5)
    #plt.scatter(img.shape[0]/2,img.shape[1]/2,s=100,marker='x')
    plt.scatter(target_aperture_params["position"][0],target_aperture_params["position"][1],s=100,marker='+')
    plt.scatter(background_aperture_params["position"][0],background_aperture_params["position"][1],s=50,marker='.',c='yellow')
    plt.xlim(target_aperture_params["position"][0]-20,target_aperture_params["position"][0]+20)
    plt.ylim(target_aperture_params["position"][1]-20,target_aperture_params["position"][1]+20)

        

    figname = 'aperture_extraction.png'
    plt.savefig(figname)
    plt.close()
    extraction_results = {
        "aperture_flux": target_flux,
        "aperture_fluxerr": target_fluxerr,
        "aperture_figure": figname
    }

    return extraction_results

