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
from shapely.geometry import Polygon
from scipy.interpolate import interp2d

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

def apply_hillshade_correction(meta_content, max_cloud_cover, crs, SCA, im, im_name, im_path, DEM_path, plot_results):
    '''Function to adjust image using by generating a hillshade model and minimizing the standard deviation of each band within the defined SCA
        INPUTS:
            - meta_content = content of the image metadata file (str array)
            - max_cloud_cover = max. cloud cover [%] - if image has cloud cover above max_cloud_cover, exits function (float)
            - crs = Coordinate Reference System (EPSG code)
            - SCA = snow-covered area where the band standard deviation is minimized (shapely.geometry.polygon.Polygon)
            - im = image (rasterio file)
            - im_name = name of image (str)
            - im_path = path to image in directory (str)
            - DEM_path = path to DEM used to generate hillshade model (str)
            - plot_results = True/False to plot results
        OUTPUTS:
            - b_corrected, g_corrected, r_corrected, nir_corrected = resulting corrected Blue, Green, Red, and NIR bands, respectively (numpy.array)'''

    print('HILLSHADE CORRECTION')
    # -----Load instrument name and cloud cover percentage from metadata
    # read the content of the file opened
    inst = meta_content[53].split('>')[1]
    if "PS2" in inst:
        inst = inst[0:3]
    elif "PSB" in inst:
        inst = inst[0:6]
    # read cloud cover percentage from the file
    cc = meta_content[148].split('>')[1]
    cc = cc.split('<')[0]
    cc = float(cc)
    # continue to next iteration if cloud cover is above max_cloud_cover
    if cc > max_cloud_cover:
        print('cloud cover exceeds max_cloud_cover... skipping')

    else:
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
            
        # -----filter image points outside the SCA
        im_x_mesh, im_y_mesh = np.meshgrid(im_x, im_y)
        b_SCA = b[np.where((im_x_mesh >= SCA.bounds[0]) & (im_x_mesh <= SCA.bounds[2]) &
                          (im_y_mesh >= SCA.bounds[1]) & (im_y_mesh <= SCA.bounds[3]))]
        g_SCA = g[np.where((im_x_mesh >= SCA.bounds[0]) & (im_x_mesh <= SCA.bounds[2]) &
                          (im_y_mesh >= SCA.bounds[1]) & (im_y_mesh <= SCA.bounds[3]))]
        r_SCA = r[np.where((im_x_mesh >= SCA.bounds[0]) & (im_x_mesh <= SCA.bounds[2]) &
                          (im_y_mesh >= SCA.bounds[1]) & (im_y_mesh <= SCA.bounds[3]))]
        nir_SCA = nir[np.where((im_x_mesh >= SCA.bounds[0]) & (im_x_mesh <= SCA.bounds[2]) &
                               (im_y_mesh >= SCA.bounds[1]) & (im_y_mesh <= SCA.bounds[3]))]
                                   
        # -----Check if image contains real values in the SCA
        if ((np.min(SCA.exterior.xy[0])>np.min(im_x))
            & (np.max(SCA.exterior.xy[0])<np.max(im_x))
            & (np.min(SCA.exterior.xy[1])>np.min(im_y))
            & (np.max(SCA.exterior.xy[1])<np.max(im_y))
            & (np.nanmean(b_SCA)>0)):
                    
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

            # -----Make directory for hillshade models (if it does not already exist in file)
            hs_path = im_path+'../hillshades/'
            if os.path.exists(hs_path)==False:
                os.mkdir(hs_path)
                print('made directory for hillshade model:'+hs_path)
            
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
                print('resampling hillshade...')
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
                print('resampled hillshade model saved to file:',hs_resamp_fn)

            # -----load resampled hillshade model
            hs_resamp = rio.open(hs_resamp_fn).read(1)
            print('resampled hillshade model loaded from file')
            # -----filter hillshade model points outside the SCA
            hs_SCA = hs_resamp[np.where((im_x_mesh >= SCA.bounds[0]) & (im_x_mesh <= SCA.bounds[2]) & (im_y_mesh >= SCA.bounds[1]) & (im_y_mesh <= SCA.bounds[3]))]

            # -----normalize hillshade model
            hs_norm = (hs_resamp - np.min(hs_resamp)) / (np.max(hs_resamp) - np.min(hs_resamp))
            hs_SCA_norm = (hs_SCA - np.min(hs_SCA)) / (np.max(hs_SCA) - np.min(hs_SCA))

            # -----plot resampled, normalized hillshade model for sanity check
    #        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,8))
    #        hs_im = ax1.imshow(hs.read(1), extent=(np.min(hs_x)/1000, np.max(hs_x)/1000, np.min(hs_y)/1000, np.max(hs_y)/1000))
    #        hsnorm_im = ax2.imshow(hs_norm, extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    #        ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='white', linewidth=2, label='SCA')
    #        fig.colorbar(hs_im, ax=ax1, shrink=0.5)
    #        fig.colorbar(hsnorm_im, ax=ax2, shrink=0.5)
    #        plt.show()
            
            # -----loop through hillshade scalar multipliers
            print('solving for optimal band scalars...')
            # define scalars to test
            hs_scalars = np.linspace(0,0.5,num=21)
            # blue
            b_SCA_mu = [] # mean
            b_SCA_sigma = [] # std
            # green
            g_SCA_mu = [] # mean
            g_SCA_sigma = [] # std
            # red
            r_SCA_mu = [] # mean
            r_SCA_sigma = [] # std
            # nir
            nir_SCA_mu = [] # mean
            nir_SCA_sigma = [] # std
            for hs_scalar in hs_scalars:
                # full image
                b_adj = b - (hs_norm * hs_scalar)
                g_adj = g - (hs_norm * hs_scalar)
                r_adj = r - (hs_norm * hs_scalar)
                nir_adj = nir - (hs_norm * hs_scalar)
                # SCA
                b_SCA_mu = b_SCA_mu + [np.nanmean(b_SCA - (hs_SCA_norm * hs_scalar))]
                b_SCA_sigma = b_SCA_sigma + [np.nanstd(b_SCA - (hs_SCA_norm * hs_scalar))]
                g_SCA_mu = g_SCA_mu + [np.nanmean(g_SCA - (hs_SCA_norm * hs_scalar))]
                g_SCA_sigma = g_SCA_sigma + [np.nanstd(g_SCA - (hs_SCA_norm * hs_scalar))]
                r_SCA_mu = r_SCA_mu + [np.nanmean(r_SCA - (hs_SCA_norm * hs_scalar))]
                r_SCA_sigma = r_SCA_sigma + [np.nanstd(r_SCA - (hs_SCA_norm * hs_scalar))]
                nir_SCA_mu = nir_SCA_mu + [np.nanmean(nir_SCA - (hs_SCA_norm * hs_scalar))]
                nir_SCA_sigma = nir_SCA_sigma + [np.nanstd(nir_SCA - (hs_SCA_norm * hs_scalar))]

            # -----Determine optimal scalar for each image band
            Ib = np.where(b_SCA_sigma==np.min(b_SCA_sigma))[0][0]
            b_scalar = hs_scalars[Ib]
            Ig = np.where(g_SCA_sigma==np.min(g_SCA_sigma))[0][0]
            g_scalar = hs_scalars[Ig]
            Ir = np.where(r_SCA_sigma==np.min(r_SCA_sigma))[0][0]
            r_scalar = hs_scalars[Ir]
            Inir = np.where(nir_SCA_sigma==np.min(nir_SCA_sigma))[0][0]
            nir_scalar = hs_scalars[Inir]
            print('Scalars:  Blue   |   Green   |   Red   |   NIR')
            print(b_scalar, g_scalar, r_scalar, nir_scalar)

            # -----Apply optimal hillshade model correction
            print('correcting bands using optimal scalar...')
            b_corrected = b - (hs_norm * hs_scalars[Ib])
            g_corrected = g - (hs_norm * hs_scalars[Ig])
            r_corrected = r - (hs_norm * hs_scalars[Ir])
            nir_corrected = nir - (hs_norm * hs_scalars[Inir])

            # replace no data values with NaN
            b_corrected[b==0] = np.nan
            g_corrected[g==0] = np.nan
            r_corrected[r==0] = np.nan
            nir_corrected[nir==0] = np.nan
                
            # -----Plot original and corrected images and band histograms
            if plot_results==True:
                fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,12), gridspec_kw={'height_ratios': [3, 1]})
                plt.rcParams.update({'font.size': 14, 'font.serif': 'Arial'})
                # original image
                ax1.imshow(np.dstack([r, g, b]),
                           extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
                ax1.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
                ax1.set_xlabel('Northing [km]')
                ax1.set_ylabel('Easting [km]')
                ax1.set_title('Original image')
                # corrected image
                ax2.imshow(np.dstack([r_corrected, g_corrected, b_corrected]),
                           extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
                ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
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
            # create output directory (if it does not already exist in file)
            out_path = im_path+'../adjusted-filtered/'
            if os.path.exists(out_path)==False:
                os.mkdir(out_path)
                print('created output directory:',out_path)
            # file name
            im_corrected_name = out_path+im_name[0:-4]+'_hs-corrected.tif'
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
            with rio.open(im_corrected_name, mode='w',**out_meta) as dst:
                dst.write_band(1,b_corrected)
                dst.write_band(2,g_corrected)
                dst.write_band(3,r_corrected)
                dst.write_band(4,nir_corrected)
            print('adjusted image saved to file:'+im_corrected_name)
                
        else:
            print('image does not contain SCA... skipping.')
            im_corrected_name = 'N/A'
            out_path = 'N/A'
        
        return im_corrected_name, out_path
            
def adjust_image_radiometry(meta_content, max_cloud_cover, im, im_name, im_path, SCA, plot_results):
    '''Function to adjust PlanetScope image band radiometry using the band values in a defined snow-covered area (SCA) and the expected surface reflectance of snow.
        INPUTS:
            -
        OUTPUTS:
            - '''
    
    print('RADIOMETRIC ADJUSTMENT')
    # -----Load instrument name and cloud cover percentage from metadata
    # read instrument name from the file
    inst = meta_content[53].split('>')[1]
    if "PS2" in inst:
        inst = inst[0:3]
    elif "PSB" in inst:
        inst = inst[0:6]
    # read cloud cover percentage from the file
    cc = meta_content[148].split('>')[1]
    cc = cc.split('<')[0]
    cc = float(cc)
    # exit if cloud cover is above 20%
    if cc > max_cloud_cover:
        print('cloud cover exceeds max_cloud_cover... skipping')
        out_path = 'N/A'
        im_corrected_name = 'N/A'
        
    else:
        # -----Define desired SR values at the bright area and darkest point for each band
        # bright area
        bright_b_adj = 0.94
        bright_g_adj = 0.95
        bright_r_adj = 0.94
        bright_nir_adj = 0.78
        # dark point
        dark_adj = 0.0
        
        # -----Define bands (blue, green, red, near infrared)
        b = im.read(1).astype(float)
        g = im.read(2).astype(float)
        r = im.read(3).astype(float)
        nir = im.read(4).astype(float)
        if (np.nanmean(b) > 1e3):
            im_scalar = 10000 # scalar multiplier for image reflectance values
            b = b / im_scalar
            g = g / im_scalar
            r = r / im_scalar
            nir = nir / im_scalar
        
        # -----Define coordinates grid
        x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
        y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
        
        # -----Check if image contains SCA
        if ((np.min(SCA.exterior.xy[0]) > np.min(x))
            & (np.max(SCA.exterior.xy[0]) < np.max(x))
            & (np.min(SCA.exterior.xy[1]) > np.min(y))
            & (np.max(SCA.exterior.xy[1]) < np.max(y))):
            
            # create vectors of image points that fall within the SCA
            x_pts = x[(x > np.min(SCA.exterior.xy[0])) & (x < np.max(SCA.exterior.xy[0]))]
            y_pts = y[(y > np.min(SCA.exterior.xy[1])) & (y < np.max(SCA.exterior.xy[1]))]
            
            # check that values are greater than zero within the SCA
            f_b = interp2d(x, y, b)
            print(np.nanmedian(f_b(x_pts, y_pts)))
            print(~np.isnan(np.nanmedian(f_b(x_pts, y_pts))))
            if (np.nanmedian(f_b(x_pts, y_pts))>0) & (~np.isnan(np.nanmedian(f_b(x_pts, y_pts)))):
                print('yes')
                # adjust SR using bright and dark points
                # band_adjusted = band*A - B
                # A = (bright_adjusted - dark_adjusted) / (bright - dark)
                # B = (dark*bright_adjusted - bright*dark_adjusted) / (bright - dark)
                # blue band
                bright_b = np.median(f_b(x_pts, y_pts)) # SR at bright point
                dark_b = np.min(b) # SR at darkest point
                A = (bright_b_adj - dark_adj) / (bright_b - dark_b)
                B = (dark_b*bright_b_adj - bright_b*dark_adj) / (bright_b - dark_b)
                b_adj = (b * A) - B
                b_adj = np.where(b==0, np.nan, b_adj) # replace no data values with nan
                # green band
                f_g = interp2d(x, y, g)
                bright_g = np.median(f_g(x_pts, y_pts)) # SR at bright point
                dark_g = np.min(g) # SR at darkest point
                A = (bright_g_adj - dark_adj) / (bright_g - dark_g)
                B = (dark_g*bright_g_adj - bright_g*dark_adj) / (bright_g - dark_g)
                g_adj = (g * A) - B
                g_adj = np.where(g==0, np.nan, g_adj) # replace no data values with nan
                # red band
                f_r = interp2d(x, y, r)
                bright_r = np.median(f_r(x_pts, y_pts)) # SR at bright point
                dark_r = np.min(r) # SR at darkest point
                A = (bright_r_adj - dark_adj) / (bright_r - dark_r)
                B = (dark_r*bright_r_adj - bright_r*dark_adj) / (bright_r - dark_r)
                r_adj = (r * A) - B
                r_adj = np.where(r==0, np.nan, r_adj) # replace no data values with nan
                # nir band
                f_nir = interp2d(x, y, nir)
                bright_nir = np.median(f_nir(x_pts, y_pts)) # SR at bright point
                dark_nir = np.min(nir) # SR at darkest point
                A = (bright_nir_adj - dark_adj) / (bright_nir - dark_nir)
                B = (dark_nir*bright_nir_adj - bright_nir*dark_adj) / (bright_nir - dark_nir)
                nir_adj = (nir * A) - B
                nir_adj = np.where(nir==0, np.nan, nir_adj) # replace no data values with nan

                # replace no data values with NaN
                b_adj[b==0] = np.nan
                g_adj[g==0] = np.nan
                r_adj[r==0] = np.nan
                nir_adj[nir==0] = np.nan
                
                # print new values at the bright and dark points to check for success
        #         f_b_adj = interp2d(x, y, b_adj)
        #         f_g_adj = interp2d(x, y, g_adj)
        #         f_r_adj = interp2d(x, y, r_adj)
        #         f_nir_adj = interp2d(x, y, nir_adj)
        #         print('    blue:',f_b_adj(bright_pt[0], bright_pt[1]))
        #         print('    green:',f_g_adj(bright_pt[0], bright_pt[1]))
        #         print('    red:',f_r_adj(bright_pt[0], bright_pt[1]))
        #         print('    nir:',f_nir_adj(bright_pt[0], bright_pt[1]))
                        
                # plot results
                if (plot_results==True):
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(16,10),gridspec_kw={'height_ratios': [3,1]})
                    plt.rcParams.update({'font.size': 12, 'font.serif': 'Arial'})
                    # original image
                    im_original = ax1.imshow(np.dstack([r, g, b]),
                                extent=(np.min(x)/1000, np.max(x)/1000, np.min(y)/1000, np.max(y)/1000))
                    ax1.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]],
                             color='black', linewidth=2, label='SCA')
                    ax1.legend()
                    ax1.set_xlabel('Easting [km]')
                    ax1.set_ylabel('Northing [km]')
                    ax1.set_title('Original image')
                    # adjusted image
                    im_adjusted = ax2.imshow(np.dstack([r_adj, g_adj, b_adj]),
                                extent=(np.min(x)/1000, np.max(x)/1000, np.min(y)/1000, np.max(y)/1000))
                    ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]],
                             color='black', linewidth=2, label='SCA')
                    ax2.set_xlabel('Easting [km]')
                    ax2.set_title('Adjusted image')
                    # histograms
                    h1_nir = ax3.hist(nir.flatten(), color='purple', bins=100, alpha=0.5, label='NIR')
                    h1_b = ax3.hist(b.flatten(), color='blue', bins=100, alpha=0.5, label='blue')
                    h1_g = ax3.hist(g.flatten(), color='green', bins=100, alpha=0.5, label='green')
                    h1_r = ax3.hist(r.flatten(), color='red', bins=100, alpha=0.5, label='red')
                    ax3.set_xlabel('Surface reflectance')
                    ax3.set_ylabel('Pixel counts')
                    ax3.grid()
                    ax3.legend(loc='right')
                    ax3.set_ylim(0,np.max([h1_nir[0][1:], h1_g[0][1:], h1_r[0][1:], h1_b[0][1:]])+5000)
                    h2_nir = ax4.hist(nir_adj.flatten(), color='purple', bins=100, alpha=0.5, label='NIR')
                    h2_b = ax4.hist(b_adj.flatten(), color='blue', bins=100, alpha=0.5, label='blue')
                    h2_g = ax4.hist(g_adj.flatten(), color='green', bins=100, alpha=0.5, label='green')
                    h2_r = ax4.hist(r_adj.flatten(), color='red', bins=100, alpha=0.5, label='red')
                    ax4.set_xlabel('Surface reflectance')
                    ax4.set_ylim(0,np.max([h1_nir[0][1:], h1_g[0][1:], h1_r[0][1:], h1_b[0][1:]])+5000)
                    ax4.grid()
                    fig.suptitle(im_name[0:8]+' '+im_name[9:11]+':'+im_name[11:13]+':'+im_name[13:15])
                    fig.tight_layout()
                    plt.show()

                # -----Save adjusted raster to file
                # create output directory (if it does not already exist in file)
                out_path = im_path+'../adjusted-filtered/'
                if os.path.exists(out_path)==False:
                    os.mkdir(out_path)
                    print('created output directory:',out_path)
                # file name
                im_adjusted_name = out_path+im_name[0:-4]+'_adj.tif'
                # metadata
                out_meta = im.meta.copy()
                out_meta.update({'driver':'GTiff',
                                 'width':b_adj.shape[1],
                                 'height':b_adj.shape[0],
                                 'count':4,
                                 'dtype':'float64',
                                 'crs':im.crs,
                                 'transform':im.transform})
                # write to file
                with rio.open(im_adjusted_name, mode='w',**out_meta) as dst:
                    dst.write_band(1,b_adj)
                    dst.write_band(2,g_adj)
                    dst.write_band(3,r_adj)
                    dst.write_band(4,nir_adj)
                print('adjusted image saved to file')

            else:
                print('no real values within SCA... skipping.')
                out_path = 'N/A'
                im_corrected_name = 'N/A'
        else:
            print('image does not contain SCA... skipping.')
            out_path = 'N/A'
            im_corrected_name = 'N/A'
                
    return im_corrected_name, out_path

