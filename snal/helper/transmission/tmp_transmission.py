

# tmporary code for constructing filter system for the selected observatory
#%%
from HHsupport_analysis import interpolate_linear
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
from HHsupport_analysis import load_filt_keys
#%%
color_key, _, _, _, _ = load_filt_keys()
filters = ['g','r','u','i']
camera_file = './KL4040_QE.csv'
filter_files = glob.glob('./SDSS_*.csv')
atm_file = glob.glob('./transmission_atm_45')[0]

# %%
plt.figure(dpi = 300)

i = 1
filter_ = filters[i]
filter_file = filter_files[0]
camera_data = ascii.read(camera_file)
filter_data = ascii.read(filter_file)
atmosp_data = Table.read(atm_file)
camera_interp = interpolate_linear(camera_data['wavelength'],camera_data['QE'], 1000, 11000, 3000)
filter_interp = interpolate_linear(filter_data['wavelength'],filter_data['transmission'], 1000, 11000, 3000)
atmosp_interp = interpolate_linear(atmosp_data['lam']*10,atmosp_data['trans'], 1000, 11000, 3000)
response_x = camera_interp[0].round(5)
response_y = (camera_interp[1]* filter_interp[1] * atmosp_interp[1]).round(5)
rpylist = []
for rp_y in response_y:
    if rp_y < 0.001:
        rpylist.append(0)
    else:
        rpylist.append(rp_y)
response_y = rpylist
plt.plot(camera_interp[0],camera_interp[1],'b',alpha = 0.5, label = 'KL4040')
plt.plot(filter_interp[0],filter_interp[1],'r',alpha = 0.5, c = color_key[filter_])
plt.plot(atmosp_interp[0],atmosp_interp[1],'k', linewidth = 0.5, alpha = 0.5, label = 'atmosphere')
plt.plot(response_x,response_y,'k',alpha = 1, c = color_key[filter_], label = filter_)
plt.legend()

plt.savefig('../transmission/KL4040_sdss_r.png')
#%%
response = Table()
response['wavelength'] = response_x
response['response'] = response_y
response.write("../transmission/KL4040_sdss_r", format = 'ascii.fixed_width')
# %%
