'''
How far is the horizon?
'''


import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 


def calc_distance(h,R):
    return R*np.tan(np.arccos(R/(R+h)))



R = 6371e3 # [m] radius of the planet
h = 10 # [m], eye level

d = calc_distance(h,R)
print(d/1e3)




''' height-distance relation

# parameters
R = 6371e3 # [m] radius of the planet
h = np.logspace(-2,5,500) # [m], eye level

d = calc_distance(h,R)


plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(h,d/1e3,color='k')
plt.text(.1,.9,'Planet radius=%.1f km'%(R/1e3),transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('Eye level (m)')
plt.ylabel('Horizon distance (km)')
plt.tight_layout()
plt.savefig('image/h-d.pdf')
plt.show()
#'''