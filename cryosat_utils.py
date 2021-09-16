import datetime as dt
from collections import defaultdict
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pyresample.geometry import AreaDefinition
from scipy.interpolate import interp1d
from skimage.util import view_as_windows

rho_water = 1026
rho_snow = 400
rho_ice = 917

dst_xmin = -3000000
dst_ymin = -3000000
dst_xmax = 3000000
dst_ymax = 3000000
dst_res = 6000
dst_width = int((dst_xmax - dst_xmin) / dst_res)
dst_height = int((dst_ymax - dst_ymin) / dst_res)
dst_extent = [dst_xmin, dst_ymin, dst_xmax, dst_ymax]
dst_proj4_string = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +rf=298.27940986765 +units=m +no_defs +type=crs'
dst_area = AreaDefinition('area_id', 'descr', 'proj_id', dst_proj4_string, dst_width, dst_height, dst_extent)
dst_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)

img_extent = [dst_xmin, dst_xmax, dst_ymax, dst_ymin]
fig_xlim = [dst_xmin+500000, dst_xmax-1500000]
fig_ylim = [dst_ymin+1700000, dst_ymax-300000]


def get_filenames(date0, date_delta0, date_delta1):
    cs2_dir = '/Data/sat/downloads/CS2/SIR_GDR/2021/01'
    cs2_files = []
    for timedelta in range(date_delta0, date_delta1+1):
        cs2_date = date0 + dt.timedelta(timedelta)
        cs2_date_str = cs2_date.strftime('%Y%m%d')
        cs2_files.extend(sorted(glob.glob(f'{cs2_dir}/CS_OFFL_SIR_GDR_2__{cs2_date_str}*D001.nc')))
    print(len(cs2_files))

    npz_name = date0.strftime('%Y%d%m')
    sit_inp_name = f'/Data/sim/antonk/sat_data_4cnn/sic_sit_def_{npz_name}.npz'
    sit_lor_name = f'/data1/antonk/dto_sit_nn/lr_input/sit_nn_{npz_name}.npy'
    sit_hir_name = f'/data1/antonk/dto_sit_nn/sar_input/sit_nn_{npz_name}.npy'

    return cs2_files, sit_inp_name, sit_lor_name, sit_hir_name

def get_sit_2d(sit_inp_name, sit_lor_name, sit_hir_name):
    return np.load(sit_inp_name)['sit'], np.load(sit_lor_name), np.load(sit_hir_name)

def read_cs2_orbits(cs2_files):
    names = ['t', 'y', 'x', 'h', 'c', 'f', 's']

    orbits = defaultdict(list)
    min_lat = 70
    min_sic = 0
    for ifile in cs2_files:
        ds = Dataset(ifile)
        freeboard_20_ku = ds['freeboard_20_ku'][:]
        lat_poca_20_ku = ds['lat_poca_20_ku'][:]
        lon_poca_20_ku = ds['lon_poca_20_ku'][:]

        time_20_ku = ds['time_20_ku'][:]
        time_cor_01 = ds['time_cor_01'][:]

        snow_depth_01 = ds['snow_depth_01'][:]    
        f = interp1d(time_cor_01.filled(np.nan), snow_depth_01.filled(np.nan), bounds_error=False)
        snow_depth_20_ku = f(time_20_ku.filled(np.nan))

        sea_ice_concentration_01 = ds['sea_ice_concentration_01'][:]
        f = interp1d(time_cor_01.filled(np.nan), sea_ice_concentration_01.filled(np.nan), bounds_error=False)
        sea_ice_concentration_20_ku = f(time_20_ku.filled(np.nan))

        # filter out southern and invalid parts of orbits
        gpi1 = (lat_poca_20_ku > min_lat) * (sea_ice_concentration_20_ku > min_sic) * (~freeboard_20_ku.mask) * (freeboard_20_ku > -1)* (freeboard_20_ku < 1)
        # compute thickness
        # https://tc.copernicus.org/preprints/tc-2021-127/
        sit = (freeboard_20_ku[gpi1] * rho_water + snow_depth_20_ku[gpi1] * rho_snow) / (rho_water - rho_ice)
        # compute X, Y coordinates in meters
        x, y, _ = dst_crs.transform_points(ccrs.PlateCarree(), lon_poca_20_ku[gpi1], lat_poca_20_ku[gpi1]).T
        # filter out errors in lon, lat positions
        for name, vector in zip(
            names,
            [time_20_ku[gpi1], y, x, sit, sea_ice_concentration_20_ku[gpi1], freeboard_20_ku[gpi1], snow_depth_20_ku[gpi1]]):
            orbits[name].append(vector)

    for n in names:
        orbits[n] = np.hstack(orbits[n])

    # compute satellite path
    t = orbits.pop('t').filled(np.nan)
    x = orbits['x']
    y = orbits['y']
    t0 = (dt.datetime(2021,1,1) - dt.datetime(2000,1,1)).total_seconds()
    spd = np.median(np.hypot(np.diff(x), np.diff(y))/np.diff(t))
    path = (t - t0) * spd / 1000 # km

    odf = pd.DataFrame.from_dict(orbits)
    odf.index = pd.TimedeltaIndex(path, "sec")        
    return odf

def plot_cs2_orbits_hist(odf):
    plt_stp = 2
    # get hi-res landmask
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
    plt.figure(figsize=(15,15))
    ax = plt.axes(projection=dst_crs)
    ax.add_feature(land_50m, zorder=0, edgecolor='black')
    gpi = odf['c'] > 0
    sct = ax.scatter(odf['x'][gpi][::plt_stp], odf['y'][gpi][::plt_stp], 10, odf['f'][gpi][::plt_stp], cmap='jet', vmin=-.1, vmax=.5)
    plt.colorbar(sct, ax=ax, shrink=0.5)
    plt.show()
    
    plt.hist(odf['f'], 100)
    plt.show()

    plt.hist2d(odf['c'][gpi], odf['f'][gpi], 30, [[0, 100], [-0.3, 0.5]])
    plt.show()

def fit_snow_ice_rho(odf, sit_cnn, plot=False):
    # smooth over 90 km, subsample to 10 km
    # for fitting ice and snow densities
    odf_filt = odf.rolling('90s').mean().resample('1s').mean().dropna()
    h_filt = odf_filt['h'].to_numpy()
    f_filt = odf_filt['f'].to_numpy()
    s_filt = odf_filt['s'].to_numpy()
    x_filt = odf_filt['x'].to_numpy()
    y_filt = odf_filt['y'].to_numpy()

    orb_cols, orb_rows = dst_area.get_array_coordinates_from_projection_coordinates(x_filt, y_filt)
    orb_rows = np.floor(orb_rows).astype(int)
    orb_cols = np.floor(orb_cols).astype(int)
    orb_sit_cnn = sit_cnn[orb_rows, orb_cols]
    gpi = np.isfinite(orb_sit_cnn*f_filt*s_filt*h_filt)*(f_filt > 0.1)

    A = np.vstack([f_filt[gpi], s_filt[gpi]]).T
    B = np.linalg.lstsq(A, orb_sit_cnn[gpi], rcond=None)[0]
    h_filt_new = np.dot(A, B)

    if plot:
        plt.plot(h_filt[gpi], orb_sit_cnn[gpi], '.', alpha=0.3, label='raw')
        plt.plot(h_filt_new, orb_sit_cnn[gpi], '.', alpha=0.3, label='tuned')
        plt.plot(orb_sit_cnn[gpi], orb_sit_cnn[gpi], 'k-')
        plt.xlim([0, 3])
        plt.ylim([0, 3])
        plt.ylabel('CNN')
        plt.xlabel('CS2')
        plt.legend()
        plt.show()

        plt.hist(h_filt_new - orb_sit_cnn[gpi], 100, [-2, 2], alpha=0.3, label='tuned')
        plt.xlabel('CS2 - CNN')
        plt.legend()
        plt.show()

    rho_ice_new = rho_water*(B[0] - 1) / B[0]
    rho_snow_new = B[1] * (rho_water - rho_ice_new)
    print(rho_ice, rho_snow)
    print(rho_ice_new, rho_snow_new)
    return rho_ice_new, rho_snow_new

def get_masked_odf(odf, rho_ice_new, rho_snow_new, smth_time='30s', min_periods=5, min_quantile=0.05):
    f_raw = odf['f'].to_numpy()
    odf_q10 = odf.rolling(smth_time, min_periods=min_periods, center=True).quantile(min_quantile)    
    mask = (f_raw > odf_q10['f']) + (f_raw > 0)
    odf_masked = odf[mask]
    f = odf_masked['f']
    s = odf_masked['s']
    odf_masked['h'] = (f * rho_water + s * rho_snow_new) / (rho_water - rho_ice_new)
    
    return odf_masked

def get_smoothed_resampled_orbits(odf, smth_time, resample_time, min_periods, t0, t1):
    if t0 is not None and t1 is not None:
        mask = (odf.index > t0) * (odf.index < t1)
        odf = odf[mask]
    
    odf = odf.rolling(smth_time, min_periods=min_periods, center=True).mean().resample(resample_time).mean().dropna()
    h = odf['h'].to_numpy()
    x = odf['x'].to_numpy()
    y = odf['y'].to_numpy()
    c, r = dst_area.get_array_coordinates_from_projection_coordinates(x, y)
    
    return c, r, h

def plot_orbit_mean(c, r, h, arrays, pi0=100, pi1=240):
    fig = plt.figure(figsize=(20,10))
    plt.plot(c[pi0:pi1], h[pi0:pi1], '.-')

    orb_rows = np.floor(r).astype(int)
    orb_cols = np.floor(c).astype(int)

    for arr in arrays:
        orb_arr = arr[orb_rows, orb_cols]
        plt.plot(c[pi0:pi1], orb_arr[pi0:pi1], '.-')

    plt.xlim([300, 500])
    plt.show()
    
def rasterize_cs2(c, r, h, sit_cnn, ws=0, stp=1, func=np.nanmean):
    orb_rows = np.floor(r).astype(int)
    orb_cols = np.floor(c).astype(int)

    grd_cs2_cnt = np.zeros(sit_cnn.shape)
    grd_cs2_sum = np.zeros(sit_cnn.shape)

    np.add.at(grd_cs2_cnt, [orb_rows, orb_cols], 1)
    np.add.at(grd_cs2_sum, [orb_rows, orb_cols], h)
    sit_cs2 = grd_cs2_sum / grd_cs2_cnt
    sit_cs2[grd_cs2_cnt == 0] = np.nan
    
    if ws > 0:
        sit_cs2_f = func(view_as_windows(sit_cs2, ws, stp), axis=(2,3))
        pad_tot = np.array(sit_cs2.shape) - np.array(sit_cs2_f.shape)
        pad_0 = np.floor(pad_tot/2).astype(int)
        pad_1 = pad_tot - pad_0
        sit_cs2_f = np.pad(sit_cs2_f, [[pad_0[0], pad_1[0]], [pad_0[1], pad_1[1]]], 'edge')
        gpi = np.isfinite(sit_cs2)
        sit_cs2[gpi] = sit_cs2_f[gpi]
    
    return sit_cs2

def compute_anomaly(arr, ws=5, stp=1, func=np.nanmean):
    func=np.nanmean
    v = view_as_windows(arr, ws, stp)
    avg = func(v, axis=(2,3))
    avg = np.pad(avg, ((2, 2), (2, 2)), 'edge')
    ano = avg - arr
    return avg, ano

def compute_precentiles(arr, p_vec, ws=5, stp=1):
    ppp = {}
    v = view_as_windows(arr, ws, stp)
    for p in p_vec:
        print(p)
        ppp[p] = np.nanpercentile(v, p, axis=(2,3))

    return ppp

def plot_maps(arrays, titles, subplot_size=5, cmap='jet', clim=(0, 3), colorbar=False, step=1):
    cols = len(arrays)
    figsize=(subplot_size*cols, subplot_size)
    fig, ax = plt.subplots(1,cols, figsize=figsize, subplot_kw=dict(projection=dst_crs))
    for i, arr in enumerate(arrays):
        img = ax[i].imshow(arr[::step, ::step], interpolation='nearest', clim=clim, extent=img_extent, cmap=cmap)
        if colorbar:
            fig.colorbar(img, ax = ax[i], shrink = 0.3)
        ax[i].set_title(titles[i])

    for a in ax.flat:
        a.add_feature(cfeature.LAND, zorder=10, edgecolor='black')
        a.set_xlim(fig_xlim)
        a.set_ylim(fig_ylim)

    plt.tight_layout()
    plt.show()

def plot_qq(arrays, titles, vlim=(-0.35, 0), common_mask=True):
    prod = np.prod(arrays, axis=0)
    gpi = np.isfinite(prod)
    qq = []
    for i, arr in enumerate(arrays):
        if common_mask:
            arr4p = arr[gpi]
        else:
            arr4p = arr
        qqq = []
        for pp in range(2,50,2):
            qqq.append(np.nanpercentile(arr4p, pp))
        qq.append(qqq)

    styles = ['k-', '.-', '.-', '.-', '.-', '.-', '.-', '.-']
    for i, qqq in enumerate(qq):
        plt.plot(qq[0], qqq, styles[i], label=titles[i])

    #plt.xlim(vlim)
    #plt.ylim(vlim)
    plt.xlabel('CS2 quantiles, m')
    plt.ylabel('Quantiles, m')
    plt.legend()
    plt.show()

def get_df_km(df):
    return df.index.days*24*60*60 + df.index.seconds + df.index.microseconds/1000000
    
    
def plot_orbit_anomaly_quantile(odf_masked, arrays, colors, labels, km0=2.8485e6, km1=2.8502e6, plot_q=0.1):
    odf_masked_km = get_df_km(odf_masked)

    gpi = (odf_masked_km > km0) * (odf_masked_km < km1) 

    odf_masked2 = odf_masked[gpi]

    odf_masked2_km = get_df_km(odf_masked2)

    h_raw = odf_masked2['h']
    x_raw = odf_masked2['x']
    y_raw = odf_masked2['y']

    c_raw, r_raw = dst_area.get_array_coordinates_from_projection_coordinates(x_raw, y_raw)

    odf_m2_pix = odf_masked2.rolling('12s').mean().resample('6s').mean().dropna()
    h_pix = odf_m2_pix['h']
    x_pix = odf_m2_pix['x']
    y_pix = odf_m2_pix['y']

    c_pix, r_pix = dst_area.get_array_coordinates_from_projection_coordinates(x_pix, y_pix)

    c_pix = np.round(c_pix).astype(int)
    r_pix = np.round(r_pix).astype(int)

    odf_m2_avg = odf_m2_pix.rolling('30s').mean()
    h_avg = odf_m2_avg['h']

    h_ano = h_avg - h_pix
    odf_m2_pix['h_ano'] = h_ano
    h_ano_p10 = odf_m2_pix.rolling('30s').quantile(plot_q)['h_ano']


    fig, ax = plt.subplots(2,1, figsize=(20,10), sharex=True)
    #plt.plot(c_raw, h_raw, 'k.')
    #plt.plot(c_pix, h_pix, 'k.')
    ax[0].plot(c_pix, h_avg, 'r.-', label='CS2')
    #ax[1].plot(c_pix, h_ano, 'r.')
    ax[1].plot(c_pix, h_ano_p10, 'r.-', label='CS2')


    for iarr, arr in enumerate(arrays):
        h_inp_pix = arr[r_pix, c_pix]
        odf_m2_pix['h_inp_pix'] = h_inp_pix
        h_inp_avg = odf_m2_pix.rolling('30s').mean()['h_inp_pix']
        h_inp_ano = h_inp_avg - h_inp_pix
        odf_m2_pix['h_inp_ano'] = h_inp_ano
        h_inp_ano_p10 = odf_m2_pix.rolling('30s').quantile(plot_q)['h_inp_ano']

        #plt.plot(c_pix, h_inp_pix, 'k.')
        ax[0].plot(c_pix, h_inp_avg, colors[iarr]+'.-', label=labels[iarr])
        #ax[1].plot(c_pix, h_inp_ano, colors[iarr]+'.')
        ax[1].plot(c_pix, h_inp_ano_p10, colors[iarr]+'.-', label=labels[iarr])

    ax[1].set_xlabel('Column coordinates')
    ax[0].set_ylabel('SIT mean, m')
    ax[1].set_ylabel('SIT anomaly P10, m')
    ax[0].legend()
    ax[1].legend()
    #ax[1].set_xlim([200, 500])
    plt.tight_layout()
