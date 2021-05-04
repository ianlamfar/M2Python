# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from Sim_Code.Objects.Particle import particle, Constants
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['pgf.texsystem'] = 'lualatex'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'font.size': 12})


class run_sim():
    def __init__(self, drop_species='decane', gas_species='air', model=2,
                 T_G=1000, rho_G=0.3529, C_p_G=1135,
                 Re_d=17, T_d=315, D=0.004):
        self.drop_species = drop_species
        self.gas_species = gas_species
        self.T_G = T_G
        self.rho_G = rho_G
        self.C_p_G = C_p_G
        self.Re_d = Re_d
        self.T_d = T_d
        self.D = D
        self.model = model

        self.c = Constants(self.drop_species, self.gas_species, self.T_G,
                           self.rho_G, self.C_p_G, self.Re_d, model=self.model)
        self.c.drop_properties()
        self.c.gas_properties()
        self.c.get_reference_conditions()
        self.c.add_drop_properties()
        self.c.add_gas_properties()
        self.c.add_properties()

        self.p = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                          D=self.D, T_d=self.T_d,
                          ODE_solver=2, coupled=2)

        self.div = self.p.get_tau()/32
        self.N = 10000

    def iter_particles(self):
        last_time = 0
        for t in range(self.N):
            if (self.p.m_d/self.p.m_d0 > 0.001 and
               self.p.T_d/self.p.T_G < 0.999):
                time1 = t * self.div
                self.p.iterate(time1 - last_time)
                last_time = time1
                # print(time1)
            else:
                break

    def plot_data(self, modellist, timelist, d2list, templist):
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        # ax1.plot(self.p.times, self.p.diameter_2_history, '--')
        # ax1.set_xlim(0)
        # ax1.set_ylim(0)
        # ax1.set_xlabel(r'$t$ ($s$)')
        # ax1.set_ylabel(r'$D^2$ ($mm^2$)')
        # ax1.set_title('Diameter Evolution of Evaporating ' +
        #               self.drop_species.title() + ' Droplet, Model '
        #               + str(self.model))

        # ax2.plot(self.p.times, self.p.temp_history, '--')
        # ax2.set_xlim(0)
        # ax2.set_ylim(self.p.T_d0)
        # ax2.set_xlabel(r'$t$ ($s$)')
        # ax2.set_ylabel(r'$T_d$ ($K$)')
        # ax2.set_title('Temperature Evolution of Evaporating ' +
        #               self.drop_species.title() + ' Droplet, Model '
        #               + str(self.model))

        # Experimental data
        t = [0, 0.35, 0.4, 0.87, 1.07, 1.26, 1.47, 1.65, 1.86, 2.05, 2.28,
             2.48, 2.67, 2.95, 3.08, 3.3, 3.48, 3.67]
        d2 = [4, 4, 4, 3.9, 3.87, 3.82, 3.75, 3.67, 3.52, 3.4, 3.28, 3.1, 2.95,
              2.65, 2.5, 2.28, 2.16, 2]
        T = [315, 330, 348, 370, 380, 387, 391, 398, 402.5, 407, 412.5, 416,
             417, 420, 421, 422, 423, 425]

        # Exp M1
        t1 = [0, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
        d21 = [4, 3.95, 3.8, 3.72, 3.45, 3.2, 2.95, 2.45, 1.8, 1.25, 0.78,
               0.25]
        T1 = [315, 355, 375, 380, 385, 390, 395, 395, 395, 395, 395, 395]

        # Exp M2
        t2 = [0, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.2]
        d22 = [4, 3.96, 3.93, 3.72, 3.7, 3.6, 3.5, 3.26, 3.08, 2.85, 2.62,
               2.42, 2.2, 2.1]
        T2 = [315, 350, 360, 362, 365, 365, 366, 370, 370, 370, 370, 370, 370,
              370]

        # Exp M3
        t3 = [0, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.2]
        d23 = [4, 3.95, 3.8, 3.75, 3.5, 3.25, 2.9, 2.3, 1.64, 1, 0.45]
        T3 = [315, 365, 380, 385, 390, 400, 402, 405, 405, 405, 405]

        # Exp M5
        t5 = [0, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.2]
        d25 = [4, 3.96, 3.93, 3.72, 3.68, 3.6, 3.4, 3.1, 2.77, 2.43, 2.1,
               1.75, 1.5, 1.38]
        T5 = [315, 365, 380, 390, 400, 415, 425, 445, 460, 475, 495, 515, 535,
              545]

        # Plot diameter first
        plt.figure(dpi=500)
        plt.plot(t, d2, "o", label="Experiment")
        plt.plot(t1, d21, "x", label="M1, Miller")
        plt.plot(t2, d22, "x", label="M2, Miller")
        plt.plot(t3, d23, "x", label="M3, Miller")
        plt.plot(t5, d25, "x", label="M5, Miller")
        for i in range(len(modellist)):
            plt.plot(timelist[i], d2list[i],
                     label=("M" + str(modellist[i]) + ", Simulation"))
            plt.xlabel(r'$t$ ($s$)')
            plt.ylabel(r'$D^2$ ($mm^2$)')
            plt.legend(fontsize=8)
            # plt.title('Diameter Evolution of Evaporating ' +
            #            self.drop_species.title())
        plt.show()
        # Plot temperature
        plt.figure(dpi=500)
        plt.plot(t, T, "o", label="Experiment")
        plt.plot(t1, T1, "x", label="M1, Miller")
        plt.plot(t2, T2, "x", label="M2, Miller")
        plt.plot(t3, T3, "x", label="M3, Miller")
        plt.plot(t5, T5, "x", label="M5, Miller")
        for i in range(len(modellist)):
            plt.plot(timelist[i], templist[i],
                     label=("M" + str(modellist[i]) + ", Simulation"))
            plt.xlabel(r'$t$ ($s$)')
            plt.ylabel(r'$T_d$ ($K$)')
            plt.legend(fontsize=8)
            # plt.title('Temperature Evolution of Evaporating ' +
            #            self.drop_species.title())
        plt.show()

    def save_data(self):
        self.file_dir = 'Sim_Code//Simulation//sim_data//'
        with open(self.file_dir + 'c_' + self.drop_species +
                  '_heat_mass_transfer_time_step_tau_32.txt', 'w') as f:
            self.p.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p.times)):
                f.write(str(self.p.times[i]) + ' ' +
                        str(self.p.temp_history[i]) + ' ' +
                        str(self.p.diameter_2_history[i]) + ' ' + '\n')
        self.p.times_temp_nd[::-1]


def run_sims(save=False):
    models = [1, 2, 3, 5]
    drop_species = ['water', 'hexane', 'decane']
    T_G = [298, 437, 1000]
    rho_G = [1.184, 0.807, 0.3529]
    C_p_G = [1007, 1020, 1141]
    Re_d = [0, 110, 17]
    T_d = [282, 281, 315]
    D = [np.sqrt(1.1)/1000, 0.00176, 0.002]
    i = 2  # decane only
    # data storage
    time_hist = []
    diameter_hist = []
    temp_hist = []

    for j in range(len(models)):
        # print(drop_species[i], "model", models[j])
        r = run_sim(drop_species[i], 'air', models[j], T_G[i],
                    rho_G[i], C_p_G[i], Re_d[i], T_d[i], D[i])
        r.iter_particles()
        time_hist.append(r.p.times)
        diameter_hist.append(r.p.diameter_2_history)
        temp_hist.append(r.p.temp_history)
        if save is True:
            r.save_data()
    r.plot_data(models, time_hist, diameter_hist, temp_hist)
