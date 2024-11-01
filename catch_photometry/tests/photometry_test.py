import numpy as np
from pytest import approx
from matplotlib import pyplot as plt

# here are functions for grabbing the data, doing background subtractions and manipulating source extractions
def get_image(url):
    from astropy.io import fits
    fits_hdu = fits.open(url)
    data = fits_hdu[0].data
    header = fits_hdu[0].header
    return data, header

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

def global_subtraction(data):
    from photutils.segmentation import detect_sources, detect_threshold
    from astropy.stats import sigma_clipped_stats, SigmaClip
    from photutils.background import SExtractorBackground as SourceExtractorBackground
    from photutils.background import Background2D, MedianBackground
    from photutils.utils import circular_footprint
    # performs a global background subtraction, masking sources using image segmentation IDs
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    
    threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=10)

    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    
    
    bkg_estimator = SourceExtractorBackground(sigma_clip)
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, mask=mask, bkg_estimator=bkg_estimator)
    
    data_sub = data-bkg.background_median

    return data_sub, bkg

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

def get_background(data):
    from photutils.segmentation import detect_sources, detect_threshold
    from astropy.stats import sigma_clipped_stats, SigmaClip
    from photutils.background import SExtractorBackground as SourceExtractorBackground
    from photutils.background import Background2D, MedianBackground
    from photutils.utils import circular_footprint
    # performs a global background subtraction, masking sources using image segmentation IDs
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    
    threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=10)

    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
     
    
    bkg_estimator = SourceExtractorBackground(sigma_clip)
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, mask=mask, bkg_estimator=bkg_estimator)
    

    return bkg
    
def test_get_background():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    bkg = get_background(data)
    assert approx(np.mean(bkg.background)) == 1569.5952576710165

#####

def id_good_sources_subtracted(data_sub,bkg):
    # uses a segmentation image to identify reliable sources that can be snapped to
    #
    # coincidentally computes baseline photometry that could be used as a quality comparison user results,
    # though this flux isn't always a good comparison as it often underestimates the source size
    source_threshold = 1.5 * bkg.background_rms
    from astropy.convolution import convolve
    from photutils.segmentation import make_2dgaussian_kernel, SourceFinder

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data_sub, kernel)
    finder = SourceFinder(npixels=5,progress_bar=False)
    segment_map = finder(convolved_data, source_threshold)
    
    
    vmax = np.percentile(np.ndarray.flatten(data_sub),99)
    vmin = np.percentile(np.ndarray.flatten(data_sub),1)
    
    # make a plot to show the background subtracted frame and the resulting segment map
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data_sub, origin='lower', cmap='Greys_r', vmin=vmin,vmax=vmax)
    ax1.set_title('Original Data')

    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
           interpolation='nearest')
    ax2.set_title('Segmentation Image')

    
    from photutils.segmentation import SourceCatalog
    cat = SourceCatalog(data_sub, segment_map, convolved_data=convolved_data)
    
    return cat

def test_id_good_sources_subtracted():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    cat = id_good_sources_subtracted(data_sub,bkg)
    assert len(cat.to_table()) == 125

#####


def id_good_sources_unsubtracted(data,bkg):
    # uses a segmentation image to identify reliable sources that can be snapped to
    #
    # coincidentally computes baseline photometry that could be used as a quality comparison user results,
    # though this flux isn't always a good comparison as it often underestimates the source size
    source_threshold = bkg.background_median + 1.5 * bkg.background_rms

    from astropy.convolution import convolve
    from photutils.segmentation import make_2dgaussian_kernel, SourceFinder

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=5, progress_bar=False)
    segment_map = finder(convolved_data, source_threshold)
    
    
    vmax = np.percentile(np.ndarray.flatten(data),99)
    vmin = np.percentile(np.ndarray.flatten(data),1)
    
    # make a plot to show the background subtracted frame and the resulting segment map
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin,vmax=vmax)
    ax1.set_title('Original Data')

    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
           interpolation='nearest')
    ax2.set_title('Segmentation Image')

    
    from photutils.segmentation import SourceCatalog
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    
    return cat

def test_id_good_sources_unsubtracted():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    data_sub, bkg = global_subtraction(data)
    cat = id_good_sources_unsubtracted(data,bkg)
    assert len(cat.to_table()) == 125

#####
    
def create_user_aperture(position,size):
    # simple placeholder function for making user-selected apertures
    from photutils.aperture import CircularAperture
    aperture = CircularAperture(position, r=size)
    return aperture

def test_create_aperture():
    aperture = create_user_aperture((11,54),12)
    assert approx(aperture.area) == 452.3893421169302

#####

def snap_to_brightest_pixel(user_point,data,radius):
    # takes in a user defined point and returns the location of the brightest pixel within radius pixels
    from photutils.centroids import centroid_quadratic, centroid_sources
    from photutils.utils import circular_footprint
    footprint = circular_footprint(radius)
    x, y = centroid_sources(data, user_point[0], user_point[1], footprint=footprint,
                            centroid_func=centroid_quadratic)
    
    target_position = np.array([x[0],y[0]])
    print(target_position)
    return target_position

def test_snap_to_brightest_pixel():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    source = snap_to_brightest_pixel([175,145],data,15)
    assert approx(source) == np.array([170.36590938, 151.69671471])

#####
    
def calc_annulus_bkg(data,position,inner_r,outer_r):
    # a function to take in an image array and position + inner/outer radii for a circular annulus
    # computes and returns background median, variance from pixels within this annulus
    # 
    # currently exports the annulus object, too, although this might not be necessary
    from photutils.aperture import CircularAnnulus
    from astropy.stats import sigma_clipped_stats, SigmaClip
    annulus_aperture = CircularAnnulus(position, r_in=inner_r, r_out=outer_r)
    annulus_mask = annulus_aperture.to_mask(method='center')
    
    annulus_data = annulus_mask.multiply(data)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]
    
    
    
    bkg_median = np.nanmedian(annulus_data_1d)
    # robust approximation via https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
    bkg_var = np.square(np.percentile(annulus_data_1d,50)-np.percentile(annulus_data_1d,16))
    
    # optional code to check the standard deviation estimate of the background
    plt.axvspan(bkg_median-np.sqrt(bkg_var),bkg_median+np.sqrt(bkg_var),alpha=0.5)
    plt.axvline(bkg_median)
    plt.hist(annulus_data_1d)
    
    
    return bkg_median, bkg_var, annulus_aperture

def test_calc_annulus_bkg():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=5, stddev=1, seed=1)
    bkg_median, bkg_var, annulus_aperture = calc_annulus_bkg(noise,(50,50),1,40)
    assert approx([bkg_median,bkg_var]) == [4.990637730701752, 0.9217670741324576]
        
def do_aperture_photometry(data,source_aperture,bkg_median, bkg_var, bkg_aperture):
    from photutils.aperture import aperture_photometry
    # takes in an image, a source aperture, and outputs from the calc_annulus_bkg function
    # 
    # returns the source flux (background subtracted, per-pixel background median) and the
    # uncertainty as defined at the quoted link
    
    # method='center' means pixels are either in or out, no interpolation to a perfect circle
    # (in other words, areas will be in whole pixels)
    aperture_mask = source_aperture.to_mask(method='center')
    aperture_data = aperture_mask.multiply(data)
    aperture_sum = np.nansum(aperture_data)
    
    background = bkg_median * source_aperture.area
    source_sum = aperture_sum-background

    
    # Using uncertainty as derived by https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
    # Setting the gain g=1, N_i = 1. Assumes data has already been converted to e-
    ### REMEMBER TO FIND THE GAINS FOR EACH SURVEY
    term1 = source_sum
    term2 = (source_aperture.area + (np.pi/2) * (np.square(source_aperture.area)/bkg_aperture.area) )*bkg_var
    source_err = np.sqrt(term1 + term2)
    
    return source_sum, source_err

#####

def test_do_aperture_photometry():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=0, stddev=1, seed=1)
    noise[50,50] = 100 # give us a "source" with flux of 100 to recover
    bkg_median, bkg_var, bkg_aperture = calc_annulus_bkg(noise,(50,50),5,10)
    source_aperture = create_user_aperture((50,50),5)
    source_sum, source_err = do_aperture_photometry(noise,source_aperture,bkg_median,bkg_var,bkg_aperture)
    assert([source_sum, source_err]) == [92.28235162232964, 15.19308011549977]


def load_thumbnail(url):
    from astropy.io import fits
    from astropy.wcs import WCS
    fits_hdu = fits.open(url)
    data = fits_hdu[0].data
    header = fits_hdu[0].header
    img_WCS = WCS(header)
    return data, header, img_WCS

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

def get_pixel_WCS(img_WCS,pixel):
    loc = img_WCS.pixel_to_world(pixel[0],pixel[1])
    return loc

def test_get_pixel_WCS():
    from photutils.datasets import make_wcs
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_pixel_WCS(wcs,[42, 57])
    assert approx([skycoord.ra.value,skycoord.dec.value]) == [197.89278975, -1.36561284]

#####

def get_WCS_pixel(img_WCS,ra_dec):
    from astropy.coordinates import SkyCoord, ICRS
    import astropy.units as u
    sky_loc = SkyCoord(ICRS(ra=ra_dec[0]*u.deg, dec=ra_dec[1]*u.deg))
    loc = img_WCS.world_to_pixel(sky_loc)
    return loc

def test_get_WCS_pixel():
    from photutils.datasets import make_wcs
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_WCS_pixel(wcs,[197.89278975, -1.36561284])
    assert approx(np.round(skycoord)) == [42.,57.]

#####  

def source_instr_mag(ap_flux,ap_fluxerr,exposure_time):
    # quick function to return instrumental magnitudes from a source flux
    # I'm really tedious about not treating magnitude uncertainties as symmetric
    instr_mag = -2.5*np.log10(ap_flux/exposure_time)
    instr_mag_hi = -2.5*np.log10((ap_flux-ap_fluxerr)/exposure_time)
    instr_mag_lo = -2.5*np.log10((ap_flux+ap_fluxerr)/exposure_time)
    
    instr_mag_hi_uncert = instr_mag_hi - instr_mag
    instr_mag_lo_uncert = instr_mag_lo - instr_mag
    
    instr_mag_array = np.array([instr_mag,instr_mag_hi_uncert,instr_mag_lo_uncert])
    
    return instr_mag_array

def test_source_instr_mag():
    mag = source_instr_mag(10,1,1)
    assert approx(mag[0] + mag[1]) == -2.38560627

#####

def calibrated_mag(instr_mag_array,zero_point,zero_point_uncert):
    # takes in the array from source_instr_mag, converts to derived magnitude,
    # propagating uncertainties from both
    calib_mag = zero_point+instr_mag_array[0]
    calib_mag_hi = np.sqrt(np.square(zero_point_uncert) + np.square(instr_mag_array[1]))
    calib_mag_lo = np.sqrt(np.square(zero_point_uncert) + np.square(instr_mag_array[2]))
    
    calib_mag_array = np.array([calib_mag,calib_mag_hi,calib_mag_lo])
    return calib_mag_array    

def test_calibrated_mag():
    calib_mag = calibrated_mag(source_instr_mag(10,1,1),22,0.5)
    assert approx(calib_mag[0] + calib_mag[1]) == 20.01291902
