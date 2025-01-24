import numpy as np
from pytest import approx
from matplotlib import pyplot as plt
from astropy.io import fits
from photutils.segmentation import detect_sources, detect_threshold, make_2dgaussian_kernel, SourceFinder, SourceCatalog, make_2dgaussian_kernel
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import SExtractorBackground as SourceExtractorBackground
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs import WCS
from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture
from photutils.centroids import centroid_quadratic, centroid_sources
from astropy import units as u

# here are functions for grabbing the data, doing background subtractions and manipulating source extractions

from ..photometry import get_image

def test_image_data():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    # test some basic values checked manually against above image values
    assert approx(data[10,10]) == 1558.4028

def test_image_header():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    # test some basic values checked manually against above header values
    assert approx(header['SECZ']) == 1.055449

#####
from ..photometry import global_subtraction

def test_global_subtraction_data():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    assert approx(np.mean(data_sub)) == 5.714418

def test_global_subtraction_bkg():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    assert approx(np.mean(bkg.background)) == 1569.5952576710165

#####

from ..photometry import get_background
    
def test_get_background():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    bkg = get_background(data)
    assert approx(np.mean(bkg.background)) == 1569.5952576710165

#####

from ..photometry import id_good_sources_subtracted

def test_id_good_sources_subtracted():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    cat = id_good_sources_subtracted(data_sub,bkg)
    assert len(cat.to_table()) == 125

#####


from ..photometry import id_good_sources_unsubtracted

def test_id_good_sources_unsubtracted():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    cat = id_good_sources_unsubtracted(data,bkg)
    assert len(cat.to_table()) == 125

#####
    
from ..photometry import create_user_aperture

def test_create_aperture():
    aperture = create_user_aperture((11,54),12)
    assert approx(aperture.area) == 452.3893421169302

#####

from ..photometry import snap_to_brightest_pixel

def test_snap_to_brightest_pixel():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    source = snap_to_brightest_pixel([175,145],data,15)
    assert approx(source) == np.array([170.36590938, 151.69671471])

#####
    
from ..photometry import calc_annulus_bkg

def test_calc_annulus_bkg():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=5, stddev=1, seed=1)
    bkg_median, bkg_var, annulus_aperture = calc_annulus_bkg(noise,(50,50),1,40)
    assert approx([bkg_median,bkg_var]) == [4.990637730701752, 0.9217670741324576]

#####
        
from ..photometry import do_aperture_photometry


def test_do_aperture_photometry():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=0, stddev=1, seed=1)
    noise[50,50] = 100 # give us a "source" with flux of 100 to recover
    bkg_median, bkg_var, bkg_aperture = calc_annulus_bkg(noise,(50,50),5,10)
    source_aperture = create_user_aperture((50,50),5)
    source_sum, source_err = do_aperture_photometry(noise,source_aperture,bkg_median,bkg_var,bkg_aperture)
    assert([source_sum, source_err]) == [92.28235162232964, 15.19308011549977]

#####

from ..photometry import load_thumbnail
    
# these tests require test.fits to be located in the test folder:
#def test_load_thumbnail_data():
#    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
#    data, header = get_image(test_url)
    # test some basic values checked manually against above image values
#    assert approx(data[10,10]) == 1558.4028

#def test_load_thumbnail_header():
#    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
#    data, header = get_image(test_url)
    # test some basic values checked manually against above header values
#    assert approx(header['SECZ']) == 1.055449

#def test_load_thumbnail_wcs():
#    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
#    data, header = get_image(test_url)
    # test some basic values checked manually against above header values
#    assert approx(header['SECZ']) == 1.055449

#####

from ..photometry import get_pixel_WCS

def test_get_pixel_WCS():
    from photutils.datasets import make_wcs
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_pixel_WCS(wcs,[42, 57])
    assert approx([skycoord.ra.value,skycoord.dec.value]) == [197.89278975, -1.36561284]

#####

from ..photometry import get_WCS_pixel

def test_get_WCS_pixel():
    from photutils.datasets import make_wcs
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_WCS_pixel(wcs,[197.89278975, -1.36561284])
    assert approx(np.round(skycoord)) == [42.,57.]

#####  

from ..photometry import source_instr_mag

def test_source_instr_mag():
    mag = source_instr_mag(10,1,1)
    assert approx(mag[0] + mag[1]) == -2.38560627

#####

from ..photometry import calibrated_mag

def test_calibrated_mag():
    calib_mag = calibrated_mag(source_instr_mag(10,1,1),22,0.5)
    assert approx(calib_mag[0] + calib_mag[1]) == 20.01291902
