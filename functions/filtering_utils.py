"""
Functions for filtering snowline time series automatically and manually
Rainey Aberle
2023
"""

import numpy as np
import pandas as pd
import glob
import rioxarray as rxr
import xarray as xr
from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --------------------------------------------------
def manual_snowline_filter_plot(sl_est_df, dataset_dict, L_im_path, PS_im_path, S2_SR_im_path, S2_TOA_im_path):
    '''
    Loop through full snowlines dataframe, plot associated image and snowline, display option to remove snowlines.

    Parameters
    ----------
    sl_est_df: pandas.DataFrame
        full, compiled dataframe of snowline CSV files
    dataset_dict: dict
        dictionary of parameters for each dataset
    L_im_path: str
        path in directory to raw Landsat images
    PS_im_path: str
        path in directory to PlanetScope image mosaics
    S2_SR_im_path: str
        path in directory to raw Sentnel-2 Surface Reflectance (SR) images
    S2_TOA_im_path: str
        path in directory to raw Sentinel-2 Top of Atmosphere reflectance (TOA) images

    Returns
    ----------
    checkboxes: list
        list of ipywidgets.widgets.widget_bool.Checkbox objects associated with each image for user input
    '''

    # -----Set the font size and checkbox size using CSS styling
    style = """
            <style>
            .my-checkbox input[type="checkbox"] {
                transform: scale(2.5); /* Adjust the scale factor as needed */
                margin-right: 20px; /* Adjust the spacing between checkbox and label as needed */
                margin-left: 20px;
            }
            .my-checkbox label {
                font-size: 24px; /* Adjust the font size as needed */
            }
            </style>
            """

    # -----Display instructions message
    print('Scroll through each snowline image and check boxes below "bad" snowlines to remove from time series.')
    print('When finished, proceed to next cell.')

    # -----Loop through snowlines
    checkboxes = [] # initalize list of heckboxes for user input
    for i in np.arange(0,len(sl_est_df)):

        print(' ')
        print(' ')

        # grab snowline coordinates
        if len(sl_est_df.iloc[i]['snowlines_coords_X']) > 2:
            sl_X = [float(x) for x in sl_est_df.iloc[i]['snowlines_coords_X'].replace('[','').replace(']','').split(', ')]
            sl_Y = [float(y) for y in sl_est_df.iloc[i]['snowlines_coords_Y'].replace('[','').replace(']','').split(', ')]
        # grab snowline date
        date = sl_est_df.iloc[i]['datetime']
        # grab snowline dataset
        dataset = sl_est_df.iloc[i]['dataset']
        print(date, dataset)

        # determine snowline image file name
        im_fn=None
        if dataset=='Landsat':
            im_fn = glob.glob(L_im_path + '*' + date.replace('-','')[0:8]+'.tif')
        elif dataset=='PlanetScope':
            im_fn = glob.glob(PS_im_path + date.replace('-','')[0:8]+'.tif')
        elif dataset=='Sentinel-2_SR':
            im_fn = glob.glob(S2_SR_im_path + date.replace('-','')[0:8] + '*.tif')
        elif dataset=='Sentinel-2_TOA':
            im_fn = glob.glob(S2_TOA_im_path + date.replace('-','')[0:8] + '*.tif')

        if im_fn:
            im_fn = im_fn[0]
        else:
            print('No image found in file')
            continue
        print(im_fn)

        # load image
        im_da = rxr.open_rasterio(im_fn)
        im_ds = im_da.to_dataset('band')
        band_names = list(dataset_dict[dataset]['refl_bands'].keys())
        im_ds = im_ds.rename({i + 1: name for i, name in enumerate(band_names)})
        im_ds = xr.where(im_ds!=dataset_dict[dataset]['no_data_value'],
                         im_ds / dataset_dict[dataset]['image_scalar'], np.nan)
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        RGB_bands = dataset_dict[dataset]['RGB_bands']
        ax.imshow(np.dstack([im_ds[RGB_bands[0]], im_ds[RGB_bands[1]], im_ds[RGB_bands[2]]]),
                  extent=(np.min(im_ds.x.data)/1e3, np.max(im_ds.x.data)/1e3,
                          np.min(im_ds.y.data)/1e3, np.max(im_ds.y.data)/1e3))
        if len(sl_est_df.iloc[i]['snowlines_coords_X']) > 2:
            ax.plot([x/1e3 for x in sl_X], [y/1e3 for y in sl_Y], '.m', markersize=2, label='snowline')
            ax.legend(loc='best')
        else:
            print('No snowline coordinates detected')
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        ax.set_title(date)
        plt.show()

        # create and display checkbox
        checkbox = widgets.Checkbox(value=False, description='Remove snowline', indent=False)
        checkbox.add_class('my-checkbox')
        display(HTML(style))
        display(checkbox)

        # add checkbox to list of checkboxes
        checkboxes += [checkbox]

    return checkboxes



# --------------------------------------------------
def fourier_series_symb(x, f, n=0):
    """
    Creates a symbolic fourier series of order 'n'.

    Parameters
    ----------
    n: float
        Order of the fourier series
    x: numpy.array
        Independent variable
    f: float
        Frequency of the fourier series

    Returns
    ----------
    series: str
        symbolic fourier series of order 'n'
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


# --------------------------------------------------
def fourier_model(c, X):
    '''
    Generates a fourier series model using the given coefficients, evaluated at the input X values.

    Parameters
    ----------
    c: numpy.array
        vector containing the coefficients for the Fourier fit
    x: numpy.array
        x-values at which to evaluate the model

    Returns
    ----------
    ymod: numpy.array
        modeled y-values values at each x-value
    '''

    if len(c): # at least a1 and b1 coefficients exist
        # grab a0 and w coefficients
        a0 = c[0]
        w = c[-1]
        # list a and b coefficients in pairs, with a coeffs in one column, b coeffs in the second column
        coeff_pairs = list(zip(*[iter(c[1:])]*2))
        # separate a and b coefficients
        a_coeffs = [y[0] for y in coeff_pairs]
        b_coeffs = [y[1] for y in coeff_pairs]
        # construct the series
        series_a = np.zeros(len(X))
        for i, x in enumerate(X): # loop through x values
            series_a[i] = np.sum([y*np.cos((i+1)*x*w) for i, y in enumerate(a_coeffs)]) # sum the a terms
        series_b = np.zeros(len(X))
        for i, x in enumerate(X): # loop through x values
            series_b[i] = np.sum([y*np.sin((i+1)*x*w) for i, y in enumerate(b_coeffs)]) # sum the a terms
        ymod = [a0+a+b for a, b in list(zip(series_a, series_b))]
    else: # only a0 coefficient exists
        ymod = a0*np.ones(len(X))

    return ymod


# --------------------------------------------------
def optimized_fourier_model(X, Y, nyears, plot_results):
    '''
    Generate a modeled fit to input data using Fourier series. First, identify the ideal number of terms for the Fourier model using 100 Monte Carlo simulations. Then, solve for the mean value for each coefficient using 500 Monte Carlo simulations.

    Parameters
    ----------
    X: numpy.array
        independent variable
    Y: numpy.array
        dependent variable
    nyears: int
        number of years (or estimated periods in your data) used to determine the range of terms to test
    plot_results: bool
        whether to plot results

    Returns
    ----------
    Y_mod: numpy.array
        modeled y values evaluated at each X-value
    '''

    # -----Identify the ideal number of terms for the Fourier model using Monte Carlo simulations
    # set up variables and parameters
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series_symb(x, f=w, n=5)}

    nmc = 100 # number of Monte Carlo simulations
    pTrain = 0.9 # percent of data to use as training
    fourier_ns = [nyears-1, nyears, nyears+1]
    print('Conducting 100 Monte Carlo simulations to determine the ideal number of model terms...')

    # loop through possible number of terms
    df_terms = pd.DataFrame(columns=['fit_minus1_err', 'fit_err', 'fit_plus1_err'])
    for i in np.arange(0,nmc):

        # split into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=pTrain, shuffle=True)
        # fit fourier curves to the training data with varying number of coeffients
        fit_minus1 = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[0])},
                    x=X_train, y=Y_train).execute()
        fit = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[1])},
                    x=X_train, y=Y_train).execute()
        fit_plus1 = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[2])},
                    x=X_train, y=Y_train).execute()
        # fit models to testing data
        Y_pred_minus1 = fit_minus1.model(x=X_test, **fit_minus1.params).y
        Y_pred = fit.model(x=X_test, **fit.params).y
        Y_pred_plus1 = fit_plus1.model(x=X_test, **fit_plus1.params).y
        # calculate error, concatenate to df
        fit_minus1_err = np.abs(Y_test - Y_pred_minus1)
        fit_err = np.abs(Y_test - Y_pred)
        fit_plus1_err = np.abs(Y_test - Y_pred_plus1)
        result = pd.DataFrame({'fit_minus1_err': fit_minus1_err,
                               'fit_err': fit_err,
                               'fit_plus1_err': fit_plus1_err})
        # add results to df
        df_terms = pd.concat([df_terms, result])

    df_terms = df_terms.reset_index(drop=True)

    # plot results
    if plot_results:
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].boxplot(fit_minus1_err);
        ax[0].set_title(str(fourier_ns[0]) + ' terms, N=' + str(len(fit_minus1_err)));
        ax[0].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[0].set_ylabel('Least absolute error')
        ax[1].boxplot(fit_err);
        ax[1].set_title(str(fourier_ns[1]) + ' terms, N=' + str(len(fit_err)));
        ax[1].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[2].boxplot(fit_plus1_err);
        ax[2].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[2].set_title(str(fourier_ns[2]) + ' terms, N=' + str(len(fit_plus1_err)));
        plt.show()

    # calculate mean error for each number of coefficients
    fit_err_mean = [np.nanmean(df_terms['fit_minus1_err']), np.nanmean(df_terms['fit_err']), np.nanmean(df_terms['fit_plus1_err'])]
    # identify best number of coefficients
    Ibest = np.argmin(fit_err_mean)
    fit_best = [fit_minus1, fit, fit_plus1][Ibest]
    fourier_n = fourier_ns[Ibest]
    print('Optimal # of model terms = ' + str(fourier_n))
    print('Mean error = +/- ' + str(np.round(fit_err_mean[Ibest])) + ' m')

    # -----Conduct Monte Carlo simulations to generate 500 Fourier models
    nmc = 500 # number of monte carlo simulations
    # initialize coefficients data frame
    cols = [val[0] for val in fit_best.params.items()]
    X_mod = np.linspace(X[0], X[-1], num=100) # points at which to evaluate the model
    Y_mod = np.zeros((nmc, len(X_mod))) # array to hold modeled Y values
    Y_mod_err = np.zeros(nmc) # array to hold error associated with each model
    print('Conducting Monte Carlo simulations to generate 500 Fourier models...')
    # loop through Monte Carlo simulations
    for i in np.arange(0,nmc):

        # split into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=pTrain, shuffle=True)

        # fit fourier model to training data
        fit = Fit({y: fourier_series_symb(x, f=w, n=fourier_n)},
                    x=X_train, y=Y_train).execute()

#        print(str(i)+ ' '+ str(len(fit.params)))

        # apply fourier model to testing data
        Y_pred = fit.model(x=X_test, **fit.params).y

        # calculate mean error
        Y_mod_err[i] = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)

        # apply the model to the full X data
        c = [c[1] for c in fit.params.items()] # coefficient values
        Y_mod[i,:] = fourier_model(c, X_mod)

    # plot results
    if plot_results:
        Y_mod_iqr = iqr(Y_mod, axis=0)
        Y_mod_median = np.nanmedian(Y_mod, axis=0)
        Y_mod_P25 = Y_mod_median - Y_mod_iqr/2
        Y_mod_P75 = Y_mod_median + Y_mod_iqr/2

        fig, ax = plt.subplots(figsize=(10,6))
        plt.rcParams.update({'font.size':14})
        ax.fill_between(X_mod, Y_mod_P25, Y_mod_P75, facecolor='blue', alpha=0.5, label='model$_{IQR}$')
        ax.plot(X_mod, np.median(Y_mod, axis=0), '.-b', linewidth=1, label='model$_{median}$')
        ax.plot(X, Y, 'ok', markersize=5, label='data')
        ax.set_ylabel('Snowline elevation [m]')
        ax.set_xlabel('Days since first observation date')
        ax.grid()
        ax.legend(loc='best')
        plt.show()

    return X_mod, Y_mod, Y_mod_err
