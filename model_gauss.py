#from gaussianPlumeModel import HUMIDIFY
import numpy as np
import sys
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import time

from gauss_func import gauss_func

import matplotlib.pyplot as plt
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



class gauss_model():
    def __init__(self):
        ###########################################################################
        # Do not change these variables                                           #
        ###########################################################################


        # SECTION 0: Definitions (normally don't modify this section)
        # view

        self.PLAN_VIEW=1 
        self.HEIGHT_SLICE=2 
        self.SURFACE_TIME=3 
        self.NO_PLOT=4 

        # wind field
        self.CONSTANT_WIND=1 
        self.FLUCTUATING_WIND=2 
        self.PREVAILING_WIND=3 

        # number of stacks
        self.ONE_STACK=1 
        self.TWO_STACKS=2 
        self.THREE_STACKS=3 

        # stability of the atmosphere
        self.CONSTANT_STABILITY=1 
        self.ANNUAL_CYCLE=2 
        self.stability_str=['Very unstable','Moderately unstable','Slightly unstable', \
            'Neutral','Moderately stable','Very stable'] 
        # Aerosol properties
        self.HUMIDIFY=2 
        self.DRY_AEROSOL=1 

        self.SODIUM_CHLORIDE=1 
        self.SULPHURIC_ACID=2 
        self.ORGANIC_ACID=3 
        self.AMMONIUM_NITRATE=4 
        self.nu=[2., 2.5, 1., 2.] 
        self.rho_s=[2160., 1840., 1500., 1725.] 
        self.Ms=[58.44e-3, 98e-3, 200e-3, 80e-3] 
        self.Mw=18e-3 


        self.dxy=100           # resolution of the model in both x and y directions
        self.dz=10
        self.domain= 5
         ###########################################################################
        self.set_input()

    
    # SECTION 1: Configuration
    # Variables can be changed by the user+++++++++++++++++++++++++++++++++++++
    def set_input(self):
        self.stab1=1  # set from 1-6
        self.stability_used=self.CONSTANT_STABILITY 


        self.output=self.PLAN_VIEW
        
        self.x_slice=26  # position (1-50) to take the slice in the x-direction
        self.y_slice=1   # position (1-50) to plot concentrations vs time

        self.stack_x=0.
        self.stack_y=0.

        self.wind_speed_int = 5.
        self.Rural = True


        self.Q=[40.]  # mass emitted per unit time
        self.H=[50.]  # stack height, m
        self.days=50           # run the model for 365 days
        self.reciver_x = 10
        self.reciver_y = 500
        self.reciver_z = 10
        #--------------------------------------------------------------------------
        self.times=np.mgrid[1:(self.days)*24+1:1]/24. 

        self.Dy=10. 
        self.Dz=10. 

  

    # SECTION 2: Act on the configuration information

    # Decide which stability profile to use
    def config_stability(self):
        if self.stability_used == self.CONSTANT_STABILITY:
        
            self.stability=self.stab1*np.ones((self.days*24,1)) 
            self.stability_str=self.stability_str[self.stab1-1] 
        elif self.tability_used == self.ANNUAL_CYCLE:

            self.stability=np.round(2.5*np.cos(self.times*2.*np.pi/(365.))+3.5) 
            self.stability_str='Annual cycle' 
        else:
            sys.exit()

    def output_prepare(self):
        self.x =  np.mgrid[-500*self.domain:500*self.domain+self.dxy:self.dxy]

        #self.x=np.mgrid[-2500:2500+self.dxy:self.dxy]  # solve on a 5 km domain
        self.y=self.x               # x-grid is same as y-grid
        # decide what kind of run to do, plan view or y-z slice, or time series
        self.nearst_x_indx = (np.abs(self.x - self.reciver_x)).argmin()
        self.nearst_y_indx = (np.abs(self.y - self.reciver_y)).argmin()
        print(self.nearst_y_indx)
        #self.nearst_z_indx = (np.abs(self.z - self.reciver_z)).argmin()
        if self.output == self.PLAN_VIEW or self.output == self.SURFACE_TIME or self.output == self.NO_PLOT:

            self.C1=np.zeros((len(self.x),len(self.y),self.days*24))  # array to store data, initialised to be zero

            [self.x,self.y]=np.meshgrid(self.x,self.y)  # x and y defined at all positions on the grid
            self.z=np.zeros(np.shape(self.x))     # z is defined to be at ground level.


        elif self.output == self.HEIGHT_SLICE:
            self.z=np.mgrid[0:500+self.dz:self.dz]        # z-grid

            self.C1=np.zeros((len(self.y),len(self.z),self.days*24))  # array to store data, initialised to be zero

            [self.y,self.z]=np.meshgrid(self.y,self.z)  # y and z defined at all positions on the grid
            self.x=self.x[self.x_slice]*np.ones(np.shape(self.y))     # x is defined to be x at x_slice       
        else:
            sys.exit()
    
    def calculate_wind(self):
        self.wind_speed=self.wind_speed_int*np.ones((self.days*24,1))  # m/s
        self.wind_dir=0.*np.ones((self.days*24,1)) 
        self.wind_dir_str='Constant wind' 

    def run(self,PB):
        self.config_stability()
        self.calculate_wind()
        self.output_prepare()
        self.C1=np.zeros((len(self.x),len(self.y),len(self.wind_dir)))
        PB.setProperty('value',0)
        for i in range(0,len(self.wind_dir)):
            PB.setValue((i/len(self.wind_dir)*100)+1)
            self.C=np.ones((len(self.x),len(self.y)))
            self.C=gauss_func(self.Q,self.wind_speed[i],self.wind_dir[i],self.x,self.y,self.z,
                self.stack_x,self.stack_y,self.H,self.Dy,self.Dz,self.stability[i],self.Rural) 
            self.C1[:,:,i]=self.C1[:,:,i]+self.C 

        self.result()


    def result(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvas
        from matplotlib.figure import Figure
        self.mean_C1 = np.mean(self.C1,axis=2)*1e6
        print(self.mean_C1[self.nearst_x_indx,self.nearst_y_indx])
        self.pointpolution = self.mean_C1[self.nearst_x_indx,self.nearst_y_indx]    

        '''self.pointpolution = self.mean_C1[int(self.reciver_x/self.dxy),int(self.reciver_y/self.dxy)]
        print(self.mean_C1.shape)
        self.maxpoint = np.unravel_index(np.argmax(self.mean_C1, axis=None), self.mean_C1.shape)'''

        #indxx = self.x[(self.reciver_x)
        #indxy = self.y.index(self.reciver_y)
        #self.pointpolution = self.mean_C1[indxx,indxy]
        #for i in range(0,len(self.wind_dir)):
            #self.C=np.ones((len(self.x),len(self.y)))
            #self.pointpolution=gauss_func(self.Q,self.wind_speed[i],self.wind_dir[i],self.reciver_x,self.reciver_y,self.reciver_z,
            #    self.stack_x,self.stack_y,self.H,self.Dy,self.Dz,self.stability[i],self.Rural) 
            #self.C1[:,:,i]=self.C1[:,:,i]+self.C 

        if self.output == self.PLAN_VIEW:
            self.fig = plt.figure()
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots()
            plt.ioff()

            self.ax = plt.pcolor(self.x,self.y,self.mean_C1, cmap='jet_r',shading='auto') 
            plt.clim((0, 1e2)) 
            plt.title(self.stability_str + '\n' + self.wind_dir_str) 
            plt.xlabel('x (metres)') 
            plt.ylabel('y (metres)') 
            cb1=plt.colorbar() 
            cb1.set_label('$m$ g m$^{-3}$') 
            self.canvas.draw()
 

        elif self.output == self.HEIGHT_SLICE:
            self.fig = plt.figure()
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots()
            plt.ioff()

            self.ax = plt.pcolor(self.y,self.z,self.mean_C1, cmap='jet_r',shading='auto')      
            plt.clim((0,1e2)) 
            plt.xlabel('y (metres)') 
            plt.ylabel('z (metres)') 
            plt.title(self.stability_str + '\n' + self.wind_dir_str) 

            cb1=plt.colorbar() 
            cb1.set_label('$m$ g m$^{-3}$') 
            self.canvas.draw()

        elif self.output == self.SURFACE_TIME:
            self.fig,(self.ax1, self.ax2) = plt.subplots(2, sharex=True, sharey=False)
            self.canvas = FigureCanvas(self.fig)
            plt.ioff()
            self.ax1.plot(self.times,1e6*np.squeeze(self.C1[self.y_slice,self.x_slice,:])) 
            try:
                self.ax1.plot(self.times,smooth(1e6*np.squeeze(self.C1[self.y_slice,self.x_slice,:]),24),'r') 
                self.ax1.legend(('Hourly mean','Daily mean'))
            except:
                sys.exit()
                
            self.ax1.set_xlabel('time (days)') 
            self.ax1.set_ylabel('Mass loading ($m$ g m$^{-3}$)') 
            self.ax1.set_title(self.stability_str +'\n' + self.wind_dir_str) 

            self.ax2.plot(self.times,self.stability) 
            self.ax2.set_xlabel('time (days)') 
            self.ax2.set_ylabel('Stability parameter') 
            '''self.fig.show()
            plt.show(block=False)
            plt.pause(0)
            plt.close()'''
            #cb1=plt.colorbar() 
            #cb1.set_label('$m$ g m$^{-3}$') 
            self.canvas.draw()

        
        elif self.output == self.NO_PLOT:
            print('Don''t plot') 
        else:
            sys.exit()



#model = gauss_model()
#model.output =model.HEIGHT_SLICE
#
#model.output =model.SURFACE_TIME

#model.run()