"""
To show the progress bars of years, months, weeks, days.
"""
import calendar
import numpy as np 
import matplotlib.pyplot as plt
from astral.sun import sun
from astral import LocationInfo, zoneinfo
from datetime import datetime, timedelta, date
from matplotlib.patches import FancyBboxPatch, BoxStyle

def get_dyn_params(now, lat, lon, timezone):
    """
    To get all the dynamic parameters (that will be needed to update the plot).

    Inputs
    ------
    now: datetime object
    lat, lon: location latitude & longitude in degrees
    timezone: string
    
    Returns
    -------
    dyn_params: dict of dynamic parameters
    """
    today = datetime(now.year, now.month, now.day)

    # number of days in a month
    n_days_in_month = calendar.monthrange(now.year, now.month)[1]

    # percents of progress bars
    ts_now = now.timestamp()
    ts_yr0 = datetime(now.year, 1, 1).timestamp()
    ts_mo0 = datetime(now.year, now.month, 1).timestamp()
    ts_wk0 = (today - timedelta(days=now.weekday())).timestamp()
    ts_da0 = today.timestamp()

    ts_yr1 = datetime(now.year+1, 1, 1).timestamp()
    ts_mo1 = datetime(now.year, now.month, n_days_in_month, 23, 59, 59).timestamp()
    ts_wk1 = (today + timedelta(days=7-now.weekday())).timestamp()
    ts_da1 = (today + timedelta(days=1)).timestamp()

    pct_yr = (ts_now - ts_yr0) / (ts_yr1 - ts_yr0)
    pct_mo = (ts_now - ts_mo0) / (ts_mo1 - ts_mo0)
    pct_wk = (ts_now - ts_wk0) / (ts_wk1 - ts_wk0)
    pct_da = (ts_now - ts_da0) / (ts_da1 - ts_da0)

    # percents of night/twilight bars 
    loc = LocationInfo("", "", "", lat, lon)
    sun_time = sun(loc.observer, date=date(now.year, now.month, now.day), tzinfo=zoneinfo.ZoneInfo(timezone))
    time_dawn = sun_time['dawn']
    time_sunrise = sun_time['sunrise']
    time_sunset = sun_time['sunset']
    time_dusk = sun_time['dusk']

    pct_dawn = (time_dawn.hour + time_dawn.minute/60 + time_dawn.second/3600) / 24
    pct_sunrise = (time_sunrise.hour + time_sunrise.minute/60 + time_sunrise.second/3600) / 24
    pct_sunset = (time_sunset.hour + time_sunset.minute/60 + time_sunset.second/3600) / 24
    pct_dusk = (time_dusk.hour + time_dusk.minute/60 + time_dusk.second/3600) / 24

    return {'year': now.year,
            'month': now.month,
            'week': now.isocalendar()[1],
            'day': now.day,
            'hour': now.hour,
            'minute': now.minute,
            'pct_yr': pct_yr,
            'pct_mo': pct_mo,
            'pct_wk': pct_wk,
            'pct_da': pct_da,
            'n_days_in_month': n_days_in_month,
            'pct_dawn': pct_dawn,
            'pct_sunrise': pct_sunrise ,
            'pct_sunset': pct_sunset,
            'pct_dusk': pct_dusk,
            }


def get_tick_coord(n, x0, x1, y0, y1):
    # n: number of ticks. x0, x1, y0, y1: coord range
    x = [x0 + np.arange(n)/n * (x1-x0)] * 2
    y = [[y0]*n, [y1]*n]

    return np.array(x), np.array(y)


def update_plot(dyn_params):    
    # progress bar
    ax_pbar_yr.set_width(dyn_params['pct_yr'] * length_bar)
    ax_pbar_mo.set_width(dyn_params['pct_mo'] * length_bar)
    ax_pbar_wk.set_width(dyn_params['pct_wk'] * length_bar)
    ax_pbar_da.set_width(dyn_params['pct_da'] * length_bar)

    # left labels 
    ax_label_yr.set_text(dyn_params['year'])
    ax_label_mo.set_text(f"{name_yr[dyn_params['month']-1]} {dyn_params['day']}")
    ax_label_wk.set_text(f"Week {dyn_params['week']}")
    ax_label_da.set_text(f"{dyn_params['hour']:02d}:{dyn_params['minute']:02d}")

    # right percents 
    ax_pct_yr.set_text(f"{100 * dyn_params['pct_yr']:.1f}%")
    ax_pct_mo.set_text(f"{100 * dyn_params['pct_mo']:.1f}%")
    ax_pct_wk.set_text(f"{100 * dyn_params['pct_wk']:.1f}%")
    ax_pct_da.set_text(f"{100 * dyn_params['pct_da']:.1f}%")

    # night/twilight bars
    ax_tbar1.set_x(x0_bar + dyn_params['pct_dawn'] * length_bar)
    ax_tbar2.set_x(x0_bar + dyn_params['pct_sunset'] * length_bar)
    ax_nbar2.set_x(x0_bar + dyn_params['pct_dusk'] * length_bar)

    ax_nbar1.set_width(dyn_params['pct_dawn'] * length_bar)
    ax_tbar1.set_width((dyn_params['pct_sunrise'] - dyn_params['pct_dawn']) * length_bar)
    ax_tbar2.set_width((dyn_params['pct_dusk'] - dyn_params['pct_sunset']) * length_bar)
    ax_nbar2.set_width((1 - dyn_params['pct_dusk']) * length_bar)

    fig.canvas.draw()

# constants -------------------------------------------------------------------

# locations
locations = {
    'Ningbo': {
        'lat': 29.8683,  # [deg]
        'lon': 121.5440,
        'timezone': 'Asia/Shanghai',
    },
}

# parameters ------------------------------------------------------------------

# location
loc_name = 'Ningbo'

# animation control
mode = 'realtime'  # 'realtime' or 'test'
tst_timedelta = .3  # [hr]
cadence = 60  # [sec], plot refresh cadence

# figure
fig_x = 10  # [inch]
fig_y = 6
c_bk = np.array([226, 226, 226])/255  # RGB background color 

# bar style
boxstyle = BoxStyle("square", pad=0)  

# static bars
x0_bar = .15
x1_bar = .88
y0_bar = .15
y1_bar = 1
ec_bar = np.array([0, 0, 0])/255
fc_bar = np.array([255, 255, 255])/255 
lw_bar = 0
width_bar = .1  # fraction

# progress bars
fc_pbar = np.array([180, 44, 44])/255  # progress bars
alpha_pbar = .6

# left labels
x_label = .02  # fraction
fontsize_label = 18

# right percents
x_pct = .9

# ticks
lw_tick = 1.2
c_tick = np.array([180, 180, 180])/255
alpha_tick = .5

# weekend bars 
fc_wkdbar = np.array([240, 227, 33])/255

# names
c_name = 'k'
name_yr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
           'Oct', 'Nov', 'Dec']
name_wk = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
name_da = np.arange(24)
fontsize_name_yr = 14
fontsize_name_mo = 12
fontsize_name_wk = 14
fontsize_name_da = 12

# night/twilight bars 
fc_nbar = np.array([28, 36, 44])/255
fc_tbar = np.array([74, 117, 183])/255
alpha_ntbar = .5

# zorders
z_bar = 1
z_wkdbar = 2
z_ntbar = 2  # night/twilight bars 
z_pbar = 3
z_tick = 4
z_name = 5

# derived static parameters ---------------------------------------------------

# static bars
width_bar *= fig_y
x0_bar *= fig_x
x1_bar *= fig_x
y0_bar *= fig_y
y1_bar *= fig_y
x_label *= fig_x
x_pct *= fig_x

length_bar = x1_bar - x0_bar
y_space_bar = (y1_bar - y0_bar) / 4

y0_bar_yr = y0_bar + 3*y_space_bar
y0_bar_mo = y0_bar + 2*y_space_bar
y0_bar_wk = y0_bar + y_space_bar
y0_bar_da = y0_bar
y1_bar_yr = y0_bar_yr + width_bar
y1_bar_mo = y0_bar_mo + width_bar
y1_bar_wk = y0_bar_wk + width_bar
y1_bar_da = y0_bar_da + width_bar

bar_yr = FancyBboxPatch([x0_bar, y0_bar_yr], length_bar, width_bar, lw=lw_bar, fc=fc_bar, ec=ec_bar, boxstyle=boxstyle, zorder=z_bar)
bar_mo = FancyBboxPatch([x0_bar, y0_bar_mo], length_bar, width_bar, lw=lw_bar, fc=fc_bar, ec=ec_bar, boxstyle=boxstyle, zorder=z_bar)
bar_wk = FancyBboxPatch([x0_bar, y0_bar_wk], length_bar, width_bar, lw=lw_bar, fc=fc_bar, ec=ec_bar, boxstyle=boxstyle, zorder=z_bar)
bar_da = FancyBboxPatch([x0_bar, y0_bar_da], length_bar, width_bar, lw=lw_bar, fc=fc_bar, ec=ec_bar, boxstyle=boxstyle, zorder=z_bar)

# y coord of texts
y_txt_yr = y0_bar_yr + width_bar/2
y_txt_mo = y0_bar_mo + width_bar/2
y_txt_wk = y0_bar_wk + width_bar/2
y_txt_da = y0_bar_da + width_bar/2

# year, week, day ticks
x_tick_yr, y_tick_yr = get_tick_coord(12, x0_bar, x1_bar, y0_bar_yr, y1_bar_yr)
x_tick_wk, y_tick_wk = get_tick_coord(7, x0_bar, x1_bar, y0_bar_wk, y1_bar_wk)
x_tick_da, y_tick_da = get_tick_coord(24, x0_bar, x1_bar, y0_bar_da, y1_bar_da)

# year, week, day names 
x_name_yr = x_tick_yr[0] + (x_tick_yr[0,1] - x_tick_yr[0,0])/2
x_name_wk = x_tick_wk[0] + (x_tick_wk[0,1] - x_tick_wk[0,0])/2
x_name_da = x_tick_da[0] + (x_tick_da[0,1] - x_tick_da[0,0])/2

# weekend bars 
wkdbar = FancyBboxPatch([x_tick_wk[0,5], y_tick_wk[0,5]], 2/7*length_bar, width_bar,lw=0, fc=fc_wkdbar, ec='none', boxstyle=boxstyle, zorder=z_wkdbar)

# derived dynamic parameters --------------------------------------------------

# dynamic parameters
dyn_params = get_dyn_params(datetime.now(), locations[loc_name]['lat'], locations[loc_name]['lon'], locations[loc_name]['timezone'])

# progress bars 
pbar_yr = FancyBboxPatch([x0_bar, y0_bar_yr], dyn_params['pct_yr']*length_bar, width_bar, lw=0, fc=fc_pbar, ec='none', boxstyle=boxstyle, alpha=alpha_pbar, zorder=z_pbar)
pbar_mo = FancyBboxPatch([x0_bar, y0_bar_mo], dyn_params['pct_mo']*length_bar, width_bar, lw=0, fc=fc_pbar, ec='none', boxstyle=boxstyle, alpha=alpha_pbar, zorder=z_pbar)
pbar_wk = FancyBboxPatch([x0_bar, y0_bar_wk], dyn_params['pct_wk']*length_bar, width_bar, lw=0, fc=fc_pbar, ec='none', boxstyle=boxstyle, alpha=alpha_pbar, zorder=z_pbar)
pbar_da = FancyBboxPatch([x0_bar, y0_bar_da], dyn_params['pct_da']*length_bar, width_bar, lw=0, fc=fc_pbar, ec='none', boxstyle=boxstyle, alpha=alpha_pbar, zorder=z_pbar)

# labels 
label_yr = f"{dyn_params['year']}"
label_mo = f"{name_yr[dyn_params['month']-1]} {dyn_params['day']}"
label_wk = f"Week {dyn_params['week']}"
label_da = f"{dyn_params['hour']:02d}:{dyn_params['minute']:02d}"

# month ticks 
x_tick_mo, y_tick_mo = get_tick_coord(dyn_params['n_days_in_month'], x0_bar, x1_bar, y0_bar_mo, y1_bar_mo)

# month names
name_mo = np.arange(1, dyn_params['n_days_in_month'] + 1)
x_name_mo = x_tick_mo[0] + (x_tick_mo[0,1] - x_tick_mo[0,0])/2

# night/twilight bars
x0_nbar1 = x0_bar
x0_tbar1 = x0_bar + dyn_params['pct_dawn'] * length_bar
x0_tbar2 = x0_bar + dyn_params['pct_sunset']*length_bar
x0_nbar2 = x0_bar + dyn_params['pct_dusk'] * length_bar

len_nbar1 = dyn_params['pct_dawn']*length_bar
len_tbar1 = (dyn_params['pct_sunrise']-dyn_params['pct_dawn'])*length_bar
len_tbar2 = (dyn_params['pct_dusk']-dyn_params['pct_sunset'])*length_bar
len_nbar2 = (1-dyn_params['pct_dusk'])*length_bar

nbar1 = FancyBboxPatch([x0_nbar1, y0_bar_da], len_nbar1, width_bar, lw=0, fc=fc_nbar, ec='none', boxstyle=boxstyle, alpha=alpha_ntbar, zorder=z_ntbar)
tbar1 = FancyBboxPatch([x0_tbar1, y0_bar_da], len_tbar1, width_bar, lw=0, fc=fc_tbar, ec='none', boxstyle=boxstyle, alpha=alpha_ntbar, zorder=z_ntbar)
tbar2 = FancyBboxPatch([x0_tbar2, y0_bar_da], len_tbar2, width_bar, lw=0, fc=fc_tbar, ec='none', boxstyle=boxstyle, alpha=alpha_ntbar, zorder=z_ntbar)
nbar2 = FancyBboxPatch([x0_nbar2, y0_bar_da], len_nbar2, width_bar, lw=0, fc=fc_nbar, ec='none', boxstyle=boxstyle, alpha=alpha_ntbar, zorder=z_ntbar)

# create plot -----------------------------------------------------------------

fig = plt.figure(figsize=(fig_x, fig_y))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# set canvas
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_facecolor(c_bk)
plt.xlim(0, fig_x)
plt.ylim(0, fig_y)

ax_bar_yr = ax.add_patch(bar_yr)
ax_bar_mo = ax.add_patch(bar_mo)
ax_bar_wk = ax.add_patch(bar_wk)
ax_bar_da = ax.add_patch(bar_da)

ax_pbar_yr = ax.add_patch(pbar_yr)
ax_pbar_mo = ax.add_patch(pbar_mo)
ax_pbar_wk = ax.add_patch(pbar_wk)
ax_pbar_da = ax.add_patch(pbar_da)

# left labels 
ax_label_yr=plt.text(x_label, y_txt_yr, label_yr, fontsize=fontsize_label, weight='bold', va='center')
ax_label_mo=plt.text(x_label, y_txt_mo, label_mo, fontsize=fontsize_label, weight='bold', va='center')
ax_label_wk=plt.text(x_label, y_txt_wk, label_wk, fontsize=fontsize_label, weight='bold', va='center')
ax_label_da=plt.text(x_label, y_txt_da, label_da, fontsize=fontsize_label, weight='bold', va='center')

# right percents 
ax_pct_yr=plt.text(x_pct, y_txt_yr, f"{100*dyn_params['pct_yr']:.1f}", fontsize=fontsize_label, weight='bold', va='center')
ax_pct_mo=plt.text(x_pct, y_txt_mo, f"{100*dyn_params['pct_mo']:.1f}", fontsize=fontsize_label, weight='bold', va='center')
ax_pct_wk=plt.text(x_pct, y_txt_wk, f"{100*dyn_params['pct_wk']:.1f}", fontsize=fontsize_label, weight='bold', va='center')
ax_pct_da=plt.text(x_pct, y_txt_da, f"{100*dyn_params['pct_da']:.1f}", fontsize=fontsize_label, weight='bold', va='center')

# ticks
plt.plot(x_tick_yr, y_tick_yr, lw=lw_tick, c=c_tick, alpha=alpha_tick, zorder=z_tick)
ax_tick_mo = plt.plot(x_tick_mo, y_tick_mo, lw=lw_tick, c=c_tick, alpha=alpha_tick, zorder=z_tick)
plt.plot(x_tick_wk, y_tick_wk, lw=lw_tick, c=c_tick, alpha=alpha_tick, zorder=z_tick)
plt.plot(x_tick_da, y_tick_da, lw=lw_tick, c=c_tick, alpha=alpha_tick, zorder=z_tick)

# weekend bars 
ax.add_patch(wkdbar)

# names
for i in range(12):
    plt.text(x_name_yr[i], y_txt_yr, name_yr[i], c=c_name, fontsize=fontsize_name_yr, weight='bold', ha='center', va='center')
ax_name_mo = []
for i in range(dyn_params['n_days_in_month']):
    ax_name_mo.append(plt.text(x_name_mo[i], y_txt_mo, name_mo[i], c=c_name, fontsize=fontsize_name_mo, weight='bold', ha='center', va='center'))
for i in range(7):
    plt.text(x_name_wk[i], y_txt_wk, name_wk[i], c=c_name, fontsize=fontsize_name_wk, weight='bold', ha='center', va='center')
for i in range(24):
    plt.text(x_name_da[i], y_txt_da, name_da[i], c=c_name, fontsize=fontsize_name_da, weight='bold', ha='center', va='center')

# night/twilight bars 
ax_nbar1 = ax.add_patch(nbar1)
ax_tbar1 = ax.add_patch(tbar1)
ax_tbar2 = ax.add_patch(tbar2)
ax_nbar2 = ax.add_patch(nbar2)

# animation loop --------------------------------------------------------------
    
loop = 0
while True:
    # determine 'now' object
    if mode == 'realtime':
        now = datetime.now()
    elif mode == 'test':
        now = datetime.now() + timedelta(hours=loop*tst_timedelta)
        loop += 1
    
    # dynamic parameter dict
    dyn_params = get_dyn_params(now, locations[loc_name]['lat'], locations[loc_name]['lon'], locations[loc_name]['timezone'])

    # update plot
    update_plot(dyn_params)

    # derived parameters
    x_tick_mo, y_tick_mo = get_tick_coord(dyn_params['n_days_in_month'], x0_bar, x1_bar, y0_bar_mo, y1_bar_mo)
    name_mo = np.arange(1, dyn_params['n_days_in_month']+1)
    x_name_mo = x_tick_mo[0] + (x_tick_mo[0,1] - x_tick_mo[0,0])/2

    # month ticks
    [i.remove() for i in ax_tick_mo]  # delete old ones
    ax_tick_mo = ax.plot(x_tick_mo, y_tick_mo, lw=lw_tick, c=c_tick, alpha=alpha_tick, zorder=z_tick)

    # month names
    [i.remove() for i in ax_name_mo]
    ax_name_mo = []
    for i in range(dyn_params['n_days_in_month']):
        ax_name_mo.append(ax.text(x_name_mo[i], y_txt_mo, name_mo[i], c=c_name, fontsize=fontsize_name_mo, weight='bold', ha='center', va='center'))

    plt.pause(cadence)
