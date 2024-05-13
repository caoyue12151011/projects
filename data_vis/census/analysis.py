'''
To analyze China's census data.

Outputs
-------
Census: N-dim array, axes = (Time, Kind, Loc, Sex, Age)

    Time: [2010, 2020]
    Kind: [城, 镇, 乡]
    Loc: 0北京 1天津 2河北 3山西 4内蒙古 5辽宁 6吉林 7黑龙江 8上海 9江苏 10浙江 
        11安徽 12福建 13江西 14山东 15河南 16湖北 17湖南 18广东 19广西 20海南 21重庆 
        22四川 23贵州 24云南 25西藏 26陕西 27甘肃 28青海 29宁夏 30新疆
    Sex: [男, 女]
    Age: 0: 0, 1: 1-4, 2: 5-9, 3: 10-14, 4: 15-19, 5: 20-24, 6: 25-29
         7: 30-34, 8: 35-39, 9: 40-44, 10: 44-49, 11: 50-54, 12: 55-59 
         13: 60-64, 14: 65-69, 15: 70-74, 16: 75-79, 17: 80-84, 
         18: 85-89, 19: 90-94, 20: 95-99, 21: 100+
'''


import numpy as np
import matplotlib.pyplot as plt


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'


def pop_pyramid(data,figsize,Age,fname,title):#
    '''
    To draw population pyramid diagram.

    Inputs
    ------
    data: ndarray of (Time, Sex, Age), population array 
    figsize: [in], (fig_x, fig_y) figure size
    Age: array of age ranges
    fname: name of the image for saving 
    title: figure title
    '''

    data /= 1e6
    data_max = data.max()

    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=.05,right=.95,top=.9,bottom=0)
    plt.plot(0,0)
    ax = plt.gca()
    # ax.set_facecolor([0,0,0,.1])

    for i in range(len(Age)):
        age1, age2 = Age[i]

        # 2010 male
        ax.add_patch(plt.Rectangle((-data[0,0,i],age1),data[0,0,i],age2-age1,
            fc='r',ec=None,lw=1,alpha=.2,zorder=10))
        # 2020 male
        ax.add_patch(plt.Rectangle((-data[1,0,i],age1),data[1,0,i],age2-age1,
            fc='none',ec='r',lw=1,zorder=20))
        # 2010 female
        ax.add_patch(plt.Rectangle((0,age1),data[0,1,i],age2-age1,
            fc='b',ec=None,lw=1,alpha=.2,zorder=10))
        # 2020 female
        ax.add_patch(plt.Rectangle((0,age1),data[1,1,i],age2-age1,
            fc='none',ec='b',lw=1,zorder=20))

    plt.text(-.8*data_max,105,'Male',fontsize=14,c='r')
    plt.text(.6*data_max,105,'Female',fontsize=14,c='b')


    # scale bar (1,2,5,10,20,50,100,200,500,1000,...) 
    Notch = np.array([1,2,5,10,20,50,100,200,500,1000])
    Suffix = ['','thousand','million','billion','trillion']

    scale0 = data_max/4 *1e6 
    dec, rem = divmod(np.log(scale0)/np.log(1000),1)

    scale_rem = Notch[np.argmin(np.abs(np.log(Notch)/np.log(1000)-rem))]
    if scale_rem==1000:
        scale_rem = 1 
        dec += 1 

    suffix = Suffix[int(dec)]
    scale = scale_rem*int(1e3**dec)
    scale_demo = scale/1e6
    
    base_x = .5*data_max 
    base_y = 85
    ax.add_patch(plt.Rectangle((base_x,base_y+6),scale_demo,5,fc='k',ec=None,
        lw=1,alpha=.4))
    ax.add_patch(plt.Rectangle((base_x,base_y),scale_demo,5,fc='none',ec='k',
        lw=1))
    plt.text(base_x+data_max*.01,base_y+7,'2010',color='w',fontsize=14)
    plt.text(base_x+data_max*.01,base_y+1,'2020',color='k',fontsize=14)
    plt.text(base_x,base_y+12,'%s %s'%(scale_rem,suffix),color='k',fontsize=14)


    # age axis 
    ages = np.arange(0,101,10) 
    for age in ages:
        plt.text(.02*data_max,age-1,'%s'%age,color='k',fontsize=12,zorder=45)
    plt.text(-.1*data_max,107,'Age',fontsize=14,c='k')


    plt.title(title,y=1.03,fontsize=20,weight='bold')
    plt.plot([0,0],[0,105],lw=1,color='k',zorder=45)
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(fname)
    plt.close()



# parameters
Time = [2010,2020]
Kind = ['City','Town','Country']
Kind_cn = ['城','镇','乡']
Loc = ['BeiJing','TianJing','HeBei','ShanXi','NeiMengGu','LiaoNing','JiLin',
    'HeiLongJiang','ShangHai','JiangSu','ZheJiang','AnHui','FuJian','JiangXi',
    'ShanDong','HeNan','HuBei','HuNan','GuangDong','GuangXi','HaiNan',
    'ChongQin','SiChuan','GuiZhou','YunNan','XiZang','ShaanXi','GanSu',
    'QingHai','NingXia','XinJiang']
Loc_cn = ['北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江','上海','江苏',
    '浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','广西','海南',
    '重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆']
Sex = ['Male','Female']
Age = [[0,1],[1,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],
    [40,45],[45,50],[50,55],[55,60],[60,65],[65,70],[70,75],[75,80],[80,85],
    [85,90],[90,95],[95,100],[100,105]] # [100,105] is actually [100,inf]

Age = np.array(Age)
Age_cen = np.mean(Age,axis=1)


# read data 
data = np.loadtxt('data/2010_city.txt') # (SexAge, Loc)
c1 = np.transpose([data[::2],data[1::2]],axes=(2,0,1)) # city (Loc, Sex, Age)
data = np.loadtxt('data/2020_city.txt') # (SexAge, Loc)
c2 = np.transpose([data[::2],data[1::2]],axes=(2,0,1))*10 # (Loc, Sex, Age)

data = np.loadtxt('data/2010_town.txt') # (SexAge, Loc)
t1 = np.transpose([data[::2],data[1::2]],axes=(2,0,1)) # town (Loc, Sex, Age)
data = np.loadtxt('data/2020_town.txt') # (SexAge, Loc)
t2 = np.transpose([data[::2],data[1::2]],axes=(2,0,1))*10 # (Loc, Sex, Age)

data = np.loadtxt('data/2010_country.txt') # (SexAge, Loc)
r1 = np.transpose([data[::2],data[1::2]],axes=(2,0,1)) # country (Loc, Sex, Age)
data = np.loadtxt('data/2020_country.txt') # (SexAge, Loc)
r2 = np.transpose([data[::2],data[1::2]],axes=(2,0,1))*10 # (Loc, Sex, Age)

Census = np.array([[c1,t1,r1],[c2,t2,r2]]) # (Time, Kind, Loc, Sex, Age)


# demo =========================================================================

''' Population pyramid 
figsize = (5,6)

# All
data = np.sum(Census,axis=(1,2))
fname = 'image/pyramid/all.pdf'
title = 'China'
pop_pyramid(data,figsize,Age,fname,title)

# by Kind
for i in range(len(Kind)):
    data = np.sum(Census[:,i],axis=1)
    fname = 'image/pyramid/%s.pdf'%Kind[i]
    title = 'China/%s'%Kind[i]
    pop_pyramid(data,figsize,Age,fname,title)

# by Loc
for i in range(len(Loc)):
    data = np.sum(Census[:,:,i],axis=1)
    fname = 'image/pyramid/%d%s.pdf'%(i,Loc[i])
    title = Loc[i]
    pop_pyramid(data,figsize,Age,fname,title)
#'''


''' sex ratio 

# parameters
figsize = (7,5)
lw = 1
alpha = .6


# by provinces
for i in range(len(Loc)):
    if not i==-1:
        continue

    plt.figure(figsize=figsize)

    # 2010 city
    ratio = c1[i,0]/c1[i,1]
    ratio_all = np.sum(c1[i,0])/np.sum(c1[i,1])
    plt.plot(Age_cen,ratio,c='b',lw=lw,ls='--',alpha=alpha,
        label='2010 city (%.2f)'%ratio_all)

    # 2010 town
    ratio = t1[i,0]/t1[i,1]
    ratio_all = np.sum(t1[i,0])/np.sum(t1[i,1])
    plt.plot(Age_cen,ratio,c='g',lw=lw,ls='--',alpha=alpha,
        label='2010 town (%.2f)'%ratio_all)

    # 2010 country
    ratio = r1[i,0]/r1[i,1]
    ratio_all = np.sum(r1[i,0])/np.sum(r1[i,1])
    plt.plot(Age_cen,ratio,c='r',lw=lw,ls='--',alpha=alpha,
        label='2010 country (%.2f)'%ratio_all)

    # 2010 all 
    male = c1[i,0]+t1[i,0]+r1[i,0]
    female = c1[i,1]+t1[i,1]+r1[i,1]
    ratio = male/female
    ratio_all = np.sum(male)/np.sum(female)
    plt.plot(Age_cen,ratio,c='k',lw=2*lw,ls='--',alpha=alpha,
        label='2010 all (%.2f)'%ratio_all)

    # 2020 city
    ratio = c2[i,0]/c2[i,1]
    ratio_all = np.sum(c2[i,0])/np.sum(c2[i,1])
    plt.plot(Age_cen,ratio,c='b',lw=lw,alpha=alpha,
        label='2020 city (%.2f)'%ratio_all)

    # 2020 town
    ratio = t2[i,0]/t2[i,1]
    ratio_all = np.sum(t2[i,0])/np.sum(t2[i,1])
    plt.plot(Age_cen,ratio,c='g',lw=lw,alpha=alpha,
        label='2020 town (%.2f)'%ratio_all)

    # 2020 country
    ratio = r2[i,0]/r2[i,1]
    ratio_all = np.sum(r2[i,0])/np.sum(r2[i,1])
    plt.plot(Age_cen,ratio,c='r',lw=lw,alpha=alpha,
        label='2020 country (%.2f)'%ratio_all)

    # 2020 all 
    male = c2[i,0]+t2[i,0]+r2[i,0]
    female = c2[i,1]+t2[i,1]+r2[i,1]
    ratio = male/female
    ratio_all = np.sum(male)/np.sum(female)
    plt.plot(Age_cen,ratio,c='k',lw=2*lw,alpha=alpha,
        label='2020 all (%.2f)'%ratio_all)

    plt.grid()
    plt.legend()
    plt.xlabel('Age (year)')
    plt.ylabel('Sex ratio')
    plt.title(Loc[i])
    plt.tight_layout()
    plt.savefig('image/sex_ratio/%s.pdf'%Loc[i])
    plt.close()


# China
plt.figure(figsize=figsize)

# 2010 city
data = np.sum(c1,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='b',lw=lw,ls='--',alpha=alpha,
    label='2010 city (%.2f)'%ratio_all)

# 2010 town
data = np.sum(t1,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='g',lw=lw,ls='--',alpha=alpha,
    label='2010 town (%.2f)'%ratio_all)

# 2010 country
data = np.sum(r1,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='r',lw=lw,ls='--',alpha=alpha,
    label='2010 country (%.2f)'%ratio_all)

# 2010 all 
data = np.sum(Census[0],axis=(0,1)) # (Sex, Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='k',lw=2*lw,ls='--',alpha=alpha,
    label='2010 all (%.2f)'%ratio_all)

# 2020 city
data = np.sum(c2,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='b',lw=lw,alpha=alpha,
    label='2020 city (%.2f)'%ratio_all)

# 2020 town
data = np.sum(t2,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='g',lw=lw,alpha=alpha,
    label='2020 town (%.2f)'%ratio_all)

# 2020 country
data = np.sum(r2,axis=0) # (Sex,Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='r',lw=lw,alpha=alpha,
    label='2020 country (%.2f)'%ratio_all)

# 2020 all 
data = np.sum(Census[1],axis=(0,1)) # (Sex, Age)
ratio = data[0]/data[1]
ratio_all = np.sum(data[0])/np.sum(data[1])
plt.plot(Age_cen,ratio,c='k',lw=2*lw,alpha=alpha,
    label='2020 all (%.2f)'%ratio_all)

plt.grid()
plt.legend()
plt.xlabel('Age (year)')
plt.ylabel('Sex ratio')
plt.title('China')
plt.tight_layout()
plt.savefig('image/sex_ratio/China.pdf')
plt.close()
#'''


''' c/t/r populations by provinces
alpha = .7 
x = np.arange(len(Loc))

for i in range(len(Time)):
    city = np.sum(Census[i,0],axis=(1,2))/1e6 # (Loc) 
    town = np.sum(Census[i,1],axis=(1,2))/1e6
    country = np.sum(Census[i,2],axis=(1,2))/1e6

    plt.figure(figsize=(10,5))
    plt.ylim(0,125)
    ax = plt.gca()
    plt.bar(x,city,fc='b',ec='none',alpha=alpha,label='City',zorder=40)
    plt.bar(x,town,bottom=city,fc='g',ec='none',alpha=alpha,label='Town',
        zorder=40)
    plt.bar(x,country,bottom=city+town,fc='r',ec='none',alpha=alpha,
        label='Country',zorder=40)
    plt.grid()
    plt.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(Loc,rotation=90,fontsize=12)
    plt.ylabel('Population (million)')
    plt.tight_layout()
    plt.savefig('image/population/%s.pdf'%Time[i])
    plt.close()
#'''


''' c/t/r populations by provinces

# parameters
alpha = 1
x = np.arange(len(Loc))


# data
city1 = np.sum(Census[0,0],axis=(1,2)) # (Loc) 
town1 = np.sum(Census[0,1],axis=(1,2))
country1 = np.sum(Census[0,2],axis=(1,2))
total1 = city1+town1+country1
r_c1 = city1/total1*100 # [%]
r_t1 = town1/total1*100 # [%]
r_r1 = country1/total1*100 # [%]
r_c1_total = np.sum(city1)/np.sum(total1)*100
r_t1_total = np.sum(town1)/np.sum(total1)*100
r_r1_total = np.sum(country1)/np.sum(total1)*100

city2 = np.sum(Census[1,0],axis=(1,2))
town2 = np.sum(Census[1,1],axis=(1,2))
country2 = np.sum(Census[1,2],axis=(1,2))
total2 = city2+town2+country2
r_c2 = city2/total2*100 # [%]
r_t2 = town2/total2*100 # [%]
r_r2 = country2/total2*100 # [%]
r_c2_total = np.sum(city2)/np.sum(total2)*100
r_t2_total = np.sum(town2)/np.sum(total2)*100
r_r2_total = np.sum(country2)/np.sum(total2)*100


# demo
plt.figure(figsize=(12,5))
ax = plt.gca()
plt.xlim(-1,37)
plt.scatter(1,1,s=0,label='2010/2020')

# 2010
plt.bar(x-.2,r_c1,width=.4,fc='b',ec='k',alpha=alpha,
    label='City (%.1f%%/%.1f%%)'%(r_c1_total,r_c2_total),zorder=40)
plt.bar(x-.2,r_t1,width=.4,bottom=r_c1,fc='g',ec='k',alpha=alpha,
    label='Town (%.1f%%/%.1f%%)'%(r_t1_total,r_t2_total),zorder=40)
plt.bar(x-.2,r_r1,width=.4,bottom=r_c1+r_t1,fc='r',ec='k',alpha=alpha,
    label='Country (%.1f%%/%.1f%%)'%(r_r1_total,r_r2_total),zorder=40)

# 2020
plt.bar(x+.2,r_c2,width=.4,fc='b',ec='k',alpha=alpha,zorder=40)
plt.bar(x+.2,r_t2,width=.4,bottom=r_c2,fc='g',ec='k',alpha=alpha)
plt.bar(x+.2,r_r2,width=.4,bottom=r_c2+r_t2,fc='r',ec='k',alpha=alpha,
    zorder=40)

# plt.grid()
plt.legend()
ax.set_xticks(x)
ax.set_xticklabels(Loc,rotation=90,fontsize=12)
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('image/population/percentage.pdf')
plt.close()
#'''






