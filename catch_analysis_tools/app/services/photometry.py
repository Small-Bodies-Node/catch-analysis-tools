from catch_analysis_tools.photometry import get_image,get_pixel_WCS,get_WCS_pixel,subpixel_centroid,create_user_aperture,do_aperture_photometry,source_instr_mag,calibrated_mag
from catch_analysis_tools.background import calc_annulus_bkg
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np

def centroid_location(filebase,target_ra,target_dec,search_radius):
    """
    Searches for source nearby to expected ephemeris location in image, assumes that astrometry solution has been rerun and that both the cutout .fits file and the redone .wcs file exist. 
    
    
    Parameters
    ----------
    filebase : string
        Base filename (without .fits or .wcs extension) taken from CATCH-generated query URL.

    target_ra : float
        Target Right Ascension ephemeris location, to be used as initial guess for the object.

    target_dec : float
        Target Declination ephemeris location, to be used as initial guess for the object.

    search_radius : float
        Radius in pixels of search area (centered on (target_ra,target_dec) ) for centroiding


    Returns
    -------
    calibrated_mag :
        array_like
                    
                    
    figname: 
        string
                    Output plot showing the default aperture + annulus extraction onto the cutout image.
    """

    
    tmp_wcs = WCS(filebase+'.wcs')
    img, header = get_image(filebase+'.fits')

    target_pix = get_WCS_pixel(tmp_wcs,target_ra,target_dec)
    target_x = target_pix[0].item()
    target_y = target_pix[1].item()

    cent_pix = subpixel_centroid([target_x,target_y],img,search_radius)
    cent_loc = get_pixel_WCS(tmp_wcs,cent_pix[0],cent_pix[1])
    
    search_results = {
        "init_guess_x":target_x,
        "init_guess_y":target_y,
        "init_guess_ra":target_ra,
        "init_guess_dec":target_dec,
        "search_radius":search_radius,
        "cent_x":cent_pix[0],
        "cent_y":cent_pix[1],
        "cent_ra":cent_loc.ra.deg,
        "cent_dec":cent_loc.dec.deg 
    }
    print(search_results)
    interval = ZScaleInterval()
    norm = ImageNormalize(img, interval=ZScaleInterval())

            
    plt.figure(figsize=(8,8))
    plt.imshow(img,norm=norm,cmap='gray_r')
    plt.scatter(img.shape[0]/2,img.shape[1]/2,s=100,marker='x')
    plt.scatter(target_x,target_y,s=100,marker='+')
    plt.scatter(cent_pix[0],cent_pix[1],s=50,marker='.',c='yellow')
    plt.xlim(target_x-20,target_x+20)
    plt.ylim(target_y-20,target_y+20)

        

    figname = filebase +'_centroid.png'
    plt.savefig(figname)
    plt.close()
    results = {
        "search_results": search_results,
        "centroid_figure": figname
    }

    return results

def default_extraction(filebase,target_ra,target_dec,search_radius,aperture_size,an_in,an_out):
    """
    Searches for source nearby to expected ephemeris location in image, assumes that astrometry solution has been rerun and that both the cutout .fits file and the redone .wcs file exist. 
    
    
    Parameters
    ----------
    filebase : string
        Base filename (without .fits or .wcs extension) taken from CATCH-generated query URL.

    target_ra : float
        Target Right Ascension ephemeris location, to be used as initial guess for the object.

    target_dec : float
        Target Declination ephemeris location, to be used as initial guess for the object.

    search_radius : float
        Radius in pixels of search area (centered on (target_ra,target_dec) ) for centroiding

    aperture_size : float
        Radius (in pixels) for source flux extraction aperture

    an_in : float
        Radius (in pixels) for inner edge of annulus for background subtraction, centered on target

    an_out : float
        Radius (in pixels) for outer edge of annulus for background subtraction, centered on target


    Returns
    -------
    calibrated_mag :
        array_like
                    
                    
    figname: 
        string
                    Output plot showing the default aperture + annulus extraction onto the cutout image.
    """


    tmp_wcs = WCS(filebase+'.wcs')
    img, header = get_image(filebase+'.fits')
    targ_loc = get_WCS_pixel(tmp_wcs,target_ra,target_dec)

    centroid_results = centroid_location(filebase,target_ra,target_dec,search_radius)
    targ_cent = np.array([centroid_results["search_results"]["cent_x"],centroid_results["search_results"]["cent_y"]])

    aperture = create_user_aperture(targ_cent,aperture_size)

    interval = ZScaleInterval()
    norm = ImageNormalize(img, interval=ZScaleInterval())

    bkg, bkg_var, annulus = calc_annulus_bkg(img,targ_cent,an_in,an_out)
            
    aperture_flux, aperture_fluxerr = do_aperture_photometry(img,aperture,bkg,bkg_var,annulus)
    instr_mag = source_instr_mag(aperture_flux,aperture_fluxerr,1)
    #cal_mag = calibrated_mag(instr_mag,header['zp'],header['zp_std']) # temporarily using the original Atlas header values

    cal_mag_array = calibrated_mag(instr_mag,header['magzpt'],header['zprms'])
    
    # could put code to filter out frames where targ_loc and targ_cent vary by more than a couple pixels here
    # (would mean star hit)
            
    plt.figure(figsize=(8,8))
    plt.imshow(img,norm=norm,cmap='gray_r')
    aperture.plot(color='blue', lw=1.5, alpha=0.5)
    annulus.plot(color='yellow',lw=1.5)
    plt.scatter(img.shape[0]/2,img.shape[1]/2,s=100,marker='x')
    plt.scatter(targ_loc[0],targ_loc[1],s=100,marker='+')
    plt.scatter(targ_cent[0],targ_cent[1],s=50,marker='.',c='yellow')
    plt.xlim(targ_loc[0]-20,targ_loc[0]+20)
    plt.ylim(targ_loc[1]-20,targ_loc[1]+20)

        

    figname = filebase +'_default.png'
    plt.savefig(figname)
    plt.close()
    extraction_results = {
        "cal_mag_array": cal_mag_array,
        "aperture_figure": figname
    }
    results = centroid_results | extraction_results

    return results

