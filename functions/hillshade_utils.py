# sunpos.py
# Modified from: https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777

import math
import rasterio as rio
import numpy as np
from pyproj import Proj, transform, Transformer
import matplotlib.pyplot as plt
import subprocess
from scipy.interpolate import interp2d
import os

def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min
    
def sunpos(when, location, refraction):

    # Extract the passed data
    year, month, day, hour, minute, second = when
    latitude, longitude = location

    # Math typing shortcuts
    rad, deg = math.radians, math.degrees
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2

    # Convert latitude and longitude to radians
    rlat = rad(latitude)
    rlon = rad(longitude)

    # Decimal hour of the day at Greenwich
    greenwichtime = hour + minute / 60 + second / 3600

    # Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )

    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873

    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768

    # Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )

    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum

    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))

    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))

    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon

    # Hour angle of the sun
    hour_ang = sidereal - rasc

    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))

    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation),
    )
    
    # Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)

    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60

    # Return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))

def apply_hillshade_correction(crs, AOI, im, im_name, hs_path, DEM_path, out_path, plot_results):

    # -----Read image bands
    im_scalar = 10000
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    # divide by im_scalar if they have not been already
    if (np.nanmean(b)>1e3):
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
    
    # -----Define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
        
    # -----filter image points outside the AOI
    im_x_mesh, im_y_mesh = np.meshgrid(im_x, im_y)
    b_AOI = b[np.where((im_x_mesh >= AOI.bounds[0]) & (im_x_mesh <= AOI.bounds[2]) &
                      (im_y_mesh >= AOI.bounds[1]) & (im_y_mesh <= AOI.bounds[3]))]
    g_AOI = g[np.where((im_x_mesh >= AOI.bounds[0]) & (im_x_mesh <= AOI.bounds[2]) &
                      (im_y_mesh >= AOI.bounds[1]) & (im_y_mesh <= AOI.bounds[3]))]
    r_AOI = r[np.where((im_x_mesh >= AOI.bounds[0]) & (im_x_mesh <= AOI.bounds[2]) &
                      (im_y_mesh >= AOI.bounds[1]) & (im_y_mesh <= AOI.bounds[3]))]
    nir_AOI = nir[np.where((im_x_mesh >= AOI.bounds[0]) & (im_x_mesh <= AOI.bounds[2]) &
                           (im_y_mesh >= AOI.bounds[1]) & (im_y_mesh <= AOI.bounds[3]))]
                               
    # -----Check if image contains real values in the AOI
    if ((np.min(AOI.exterior.xy[0])>np.min(im_x))
        & (np.max(AOI.exterior.xy[0])<np.max(im_x))
        & (np.min(AOI.exterior.xy[1])>np.min(im_y))
        & (np.max(AOI.exterior.xy[1])<np.max(im_y))
        & (np.nanmean(b_AOI)>0)):
        
        # -----Extract image information for sun position calculation
        # location: grab center image coordinate, convert to lat lon
        xmid = ((im.bounds.right - im.bounds.left)/2 + im.bounds.left)
        ymid = ((im.bounds.top - im.bounds.bottom)/2 + im.bounds.bottom)
        transformer = Transformer.from_crs("epsg:"+str(crs), "epsg:4326")
        location = transformer.transform(xmid, ymid)
        # when: year, month, day, hour, minute, second
        when = (float(im_name[0:4]), float(im_name[4:6]), float(im_name[6:8]),
                float(im_name[9:11]), float(im_name[11:13]), float(im_name[13:15]))
        # sun azimuth and elevation
        azimuth, elevation = sunpos(when, location, refraction=1)
        print('location:',location)
        print('when:',when)
        print('azimuth: ',azimuth)
        print('elevation:',elevation)

        # -----Create hillshade model (if it does not already exist in file)
        hs_fn = hs_path+str(azimuth)+'-az_'+str(elevation)+'-z_hillshade.tif'
        if os.path.exists(hs_fn):
            print('hillshade model already exists in directory, loading...')
        else:
            print('creating hillshade model...')
            # construct the gdal_merge command
            # modified from: https://github.com/clhenrick/gdal_hillshade_tutorial
            # gdaldem hillshade -az aximuth -z elevation dem.tif hillshade.tif
            cmd = 'gdaldem hillshade -az '+str(azimuth)+' -z '+str(elevation)+' '+str(DEM_path)+' '+hs_fn
            # run the command
            p = subprocess.run(cmd, shell=True, capture_output=True)
            print(p)

        # -----load hillshade model from file
        hs = rio.open(hs_fn)
        print('hillshade model loaded from file...')
        # coordinates
        hs_x = np.linspace(hs.bounds.left, hs.bounds.right, num=np.shape(hs.read(1))[1])
        hs_y = np.linspace(hs.bounds.top, hs.bounds.bottom, num=np.shape(hs.read(1))[0])

        # -----Resample hillshade to image coordinates
        # resampled hillshade file name
        hs_resamp_fn = hs_path+str(azimuth)+'-az_'+str(elevation)+'-z_hillshade_resamp.tif'
        # check if file already exists in directory
        if os.path.exists(hs_resamp_fn):
            print('resampled hillshade model already exists in directory, loading...')
        # resample if it doesn't exist
        else:
            print('    resampling hillshade...')
            # create interpolation object
            f = interp2d(hs_x, hs_y, hs.read(1))
            hs_resamp = f(im_x, im_y)
            hs_resamp = np.flipud(hs_resamp)
            # save to file
            with rio.open(hs_resamp_fn,'w',
                          driver='GTiff',
                          height=hs_resamp.shape[0],
                          width=hs_resamp.shape[1],
                          dtype=hs_resamp.dtype,
                          count=1,
                          crs=im.crs,
                          transform=im.transform) as dst:
                dst.write(hs_resamp, 1)
            print('    resampled hillshade model saved to file:',hs_resamp_fn)

        # -----load resampled hillshade model
        hs_resamp = rio.open(hs_resamp_fn).read(1)
        print('resampled hillshade model loaded from file')
        # -----filter hillshade model points outside the AOI
        hs_AOI = hs_resamp[np.where((im_x_mesh >= AOI.bounds[0]) & (im_x_mesh <= AOI.bounds[2]) &
                                    (im_y_mesh >= AOI.bounds[1]) & (im_y_mesh <= AOI.bounds[3]))]

        # -----normalize hillshade model
        hs_norm = (hs_resamp - np.min(hs_resamp)) / (np.max(hs_resamp) - np.min(hs_resamp))
        hs_AOI_norm = (hs_AOI - np.min(hs_AOI)) / (np.max(hs_AOI) - np.min(hs_AOI))

        # -----plot resampled, normalized hillshade model for sanity check
#        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,8))
#        hs_im = ax1.imshow(hs.read(1), extent=(np.min(hs_x)/1000, np.max(hs_x)/1000, np.min(hs_y)/1000, np.max(hs_y)/1000))
#        hsnorm_im = ax2.imshow(hs_norm, extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
#        ax2.plot([x/1000 for x in AOI.exterior.xy[0]], [y/1000 for y in AOI.exterior.xy[1]], color='white', linewidth=2, label='AOI')
#        fig.colorbar(hs_im, ax=ax1, shrink=0.5)
#        fig.colorbar(hsnorm_im, ax=ax2, shrink=0.5)
#        plt.show()
        
        # -----loop through hillshade scalar multipliers
        print('solving for optimal band scalars...')
        # define scalars to test
        hs_scalars = np.linspace(0,0.5,num=21)
        # blue
        b_AOI_mu = [] # mean
        b_AOI_sigma = [] # std
        # green
        g_AOI_mu = [] # mean
        g_AOI_sigma = [] # std
        # red
        r_AOI_mu = [] # mean
        r_AOI_sigma = [] # std
        # nir
        nir_AOI_mu = [] # mean
        nir_AOI_sigma = [] # std
        for hs_scalar in hs_scalars:
            # full image
            b_adj = b - (hs_norm * hs_scalar)
            g_adj = g - (hs_norm * hs_scalar)
            r_adj = r - (hs_norm * hs_scalar)
            nir_adj = nir - (hs_norm * hs_scalar)
            # AOI
            b_AOI_mu = b_AOI_mu + [np.nanmean(b_AOI - (hs_AOI_norm * hs_scalar))]
            b_AOI_sigma = b_AOI_sigma + [np.nanstd(b_AOI - (hs_AOI_norm * hs_scalar))]
            g_AOI_mu = g_AOI_mu + [np.nanmean(g_AOI - (hs_AOI_norm * hs_scalar))]
            g_AOI_sigma = g_AOI_sigma + [np.nanstd(g_AOI - (hs_AOI_norm * hs_scalar))]
            r_AOI_mu = r_AOI_mu + [np.nanmean(r_AOI - (hs_AOI_norm * hs_scalar))]
            r_AOI_sigma = r_AOI_sigma + [np.nanstd(r_AOI - (hs_AOI_norm * hs_scalar))]
            nir_AOI_mu = nir_AOI_mu + [np.nanmean(nir_AOI - (hs_AOI_norm * hs_scalar))]
            nir_AOI_sigma = nir_AOI_sigma + [np.nanstd(nir_AOI - (hs_AOI_norm * hs_scalar))]

        # -----Determine optimal scalar for each image band
        Ib = np.where(b_AOI_sigma==np.min(b_AOI_sigma))[0][0]
        b_scalar = hs_scalars[Ib]
        Ig = np.where(g_AOI_sigma==np.min(g_AOI_sigma))[0][0]
        g_scalar = hs_scalars[Ig]
        Ir = np.where(r_AOI_sigma==np.min(r_AOI_sigma))[0][0]
        r_scalar = hs_scalars[Ir]
        Inir = np.where(nir_AOI_sigma==np.min(nir_AOI_sigma))[0][0]
        nir_scalar = hs_scalars[Inir]
        print(b_scalar, g_scalar, r_scalar, nir_scalar)

        # -----Apply optimal hillshade model correction
        print('correcting bands using optimal scalar...')
        b_corrected = b - (hs_norm * hs_scalars[Ib])
        g_corrected = g - (hs_norm * hs_scalars[Ig])
        r_corrected = r - (hs_norm * hs_scalars[Ir])
        nir_corrected = nir - (hs_norm * hs_scalars[Inir])

        # -----Plot original and corrected images and band histograms
        if plot_results==True:
            fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,12), gridspec_kw={'height_ratios': [3, 1]})
            plt.rcParams.update({'font.size': 14, 'font.serif': 'Arial'})
            # original image
            ax1.imshow(np.dstack([r, g, b]),
                       extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
            ax1.plot([x/1000 for x in AOI.exterior.xy[0]], [y/1000 for y in AOI.exterior.xy[1]], color='black', linewidth=2, label='AOI')
            ax1.set_xlabel('Northing [km]')
            ax1.set_ylabel('Easting [km]')
            ax1.set_title('Original image')
            # corrected image
            ax2.imshow(np.dstack([r_corrected, g_corrected, b_corrected]),
                       extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
            ax2.plot([x/1000 for x in AOI.exterior.xy[0]], [y/1000 for y in AOI.exterior.xy[1]], color='black', linewidth=2, label='AOI')
            ax2.set_xlabel('Northing [km]')
            ax2.set_title('Corrected image')
            # band histograms
            ax3.hist(nir[nir>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
            ax3.hist(b[b>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
            ax3.hist(g[g>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
            ax3.hist(r[r>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
            ax3.set_xlabel('Surface reflectance')
            ax3.set_ylabel('Pixel counts')
            ax3.grid()
            ax3.legend()
            ax4.hist(nir_corrected[nir_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
            ax4.hist(b_corrected[b_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
            ax4.hist(g_corrected[g_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
            ax4.hist(r_corrected[r_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
            ax4.set_xlabel('Surface reflectance')
            ax4.grid()
            fig1.tight_layout()
            plt.show()

        # -----save hillshade-corrected image to file
        # file name
        fn = out_path+im_name[0:-4]+'_hs-corrected.tif'
        # metadata
        out_meta = im.meta.copy()
        out_meta.update({'driver':'GTiff',
                         'width':b_corrected.shape[1],
                         'height':b_corrected.shape[0],
                         'count':4,
                         'dtype':'float64',
                         'crs':im.crs,
                         'transform':im.transform})
        # write to file
        with rio.open(fn, mode='w',**out_meta) as dst:
            dst.write_band(1,b_corrected)
            dst.write_band(2,g_corrected)
            dst.write_band(3,r_corrected)
            dst.write_band(4,nir_corrected)
        print('adjusted image saved to file.')

    else:
        print('image does not contain AOI... skipping.')
