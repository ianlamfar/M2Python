# -*- coding: utf-8 -*-
# Conventions:
# Coupled: (0)decoupled mass ODE (1)decoupled temp ODE from mass ODE (2)coupled
# ODE Solver: (0)analytic (1)forward (2)backward (3)modified (4)runge-kutta
# Models: (1)Classic (2)Abramazon-Sirignano (3)Mass analogy Ia (4)M.a.Ib
# Models: (5)M.a.IIa (6)M.a.IIb (7)Langumuir-Knusden I (8)LK II

# Errors: M4/6 not ok, on hold
# Complete M2 first

# Y_F = Y_s_eq, Y_F_inf = Y_G

from scipy import optimize as opt
import numpy as np
np.seterr(all='raise')


class Constants():
    def __init__(self, drop_species='water', gas_species='air', T_G=298,
                 rho_G=1.184, C_p_G=1007, Re_d=0, model=1):
        self.drop_species = drop_species
        self.gas_species = gas_species
        self.T_G = T_G
        self.rho_G = rho_G
        self.C_p_G = C_p_G
        self.Re_d = Re_d
        self.model = model

    def get_reference_conditions(self):
        self.T_WB = (137 * ((self.T_B/373.15)**(0.68)) *
                     np.log10(self.T_G)) - 45
        self.T_R = self.T_WB

    def drop_properties(self):
        if self.drop_species == 'water':
            self.W_V = 18.015
            self.T_B = 373.15
            self.rho_d = 997
            self.C_L = 4184

        if self.drop_species == 'decane':
            self.W_V = 142
            self.T_B = 447.7
            self.rho_d = 642
            self.C_L = 2520.5

        if self.drop_species == 'hexane':
            self.W_V = 86.178
            self.T_B = 344.6
            self.rho_d = 664
            self.C_L = 2302

    def gas_properties(self):
        if self.gas_species == 'air':
            self.W_G = 28.97
            self.theta_2 = self.W_G/self.W_V
            self.P_atm = 101325
            self.R_bar = 8314.5
            self.R = 287
            self.Y_G = 0

    def add_drop_properties(self):
        if self.drop_species == 'water':
            self.L_V = (2.257*(10**6) + (2.595*(10**3) * (373.15-self.T_R)))
            self.C_p_V = (8137 - 37.34*self.T_R + (0.07482*(self.T_R**2)) -
                          (4.956e-5*(self.T_R**3)))

        if self.drop_species == 'decane':
            self.L_V = (3.958*10**(4))*((619-self.T_R)**(0.38))
            self.T_star = self.T_R/1000
            if (self.T_star) <= 0.8:
                self.C_p_V = (106.6 + 5765*self.T_star - 1675*(self.T_star**2)
                              + 473.1*(self.T_star**3))
            else:
                self.C_p_V = (411.1 + 5460*self.T_star - 2483*(self.T_star**2)
                              + 422.9*(self.T_star**3))

        if self.drop_species == 'hexane':
            self.L_V = (5.14787e5)*((1-(self.T_R/512))**(0.3861))
            self.C_p_V = (-51.31 + 6.767*self.T_R - (3.626e-3*(self.T_R**2)))

    def add_gas_properties(self):
        self.P_G = self.rho_G*self.R*self.T_R
        self.mu_G = (6.109*10**(-6) + (4.604*10**(-8)*self.T_R) -
                     (1.05*10**(-11)*self.T_R**2))
        self.Pr_G = (0.815 - ((4.958*10**(-4))*self.T_R) +
                     ((4.514*10**(-7))*self.T_R**2))
        self.Sc_G = self.Pr_G  # Schmidt number for unity Lewis number

    def add_properties(self):
        self.theta_1 = self.C_p_G/self.C_L

        # Initialise H_deltaT for M1-7
        if self.model != 8:
            self.H_deltaT = 0

        # Initialise f2 for M1/3/5
        if self.model == 1 or self.model == 3 or self.model == 5:
            self.f_2 = 1


class particle(Constants):
    def __init__(self, Constants, position, velocity, D=np.sqrt(1.1)/1000,
                 T_d=282, coupled=2, ODE_solver=2,
                 get_vel_fluid=None, get_gravity=None):
        self.coupled = coupled

        self.pos = np.array(position)
        self.vel = np.array(velocity)

        if callable(get_vel_fluid) and len(get_vel_fluid(self)) == 3:
            self.get_vel_fluid = get_vel_fluid
        elif get_vel_fluid is not None:
            print("get_vel_fluid is not a valid function.")
        else:
            self.get_vel_fluid = lambda dummy: [0, 0.001, 0]

        if callable(get_gravity) and len(get_gravity(self)) == 3:
            self.get_gravity = get_gravity
        elif get_gravity is not None:
            print("get_gravity is not a valid function.")
        else:
            self.get_gravity = lambda delta_t: np.array([0, 0, 0])

        self.model = Constants.model
        self.drop_species = Constants.drop_species
        self.T_B = Constants.T_B
        self.rho_d = Constants.rho_d
        self.C_p_V = Constants.C_p_V
        self.L_V = Constants.L_V
        self.W_V = Constants.W_V
        self.C_L = Constants.C_L

        self.theta_2 = Constants.theta_2
        self.P_atm = Constants.P_atm
        self.R_bar = Constants.R_bar
        self.R = Constants.R
        self.Y_G = Constants.Y_G
        self.T_G = Constants.T_G
        self.P_G = Constants.P_G
        self.mu_G = Constants.mu_G
        self.Pr_G = Constants.Pr_G
        self.Sc_G = Constants.Sc_G
        self.rho_G = Constants.rho_G
        self.C_p_G = Constants.C_p_G

        self.P_atm = Constants.P_atm
        self.Re_d = Constants.Re_d  # (self.rho_G*self.u_s)/(self.mu_G)

        self.time = 0
        self.T_d0 = T_d
        self.T_d = np.array(T_d)

        # This is only for the initial value, M2/7/8 initials???
        if self.model == 1 or self.model == 3 or self.model == 5:
            self.f_2 = Constants.f_2
        elif self.model == 4 or self.model == 6:
            self.B_T = (self.T_G - self.T_d) * self.C_p_V / self.L_V
            self.f_2 = (1 - self.B_T)**-1
            # print(self.T_G, self.T_d, self.C_p_V, self.L_V, self.B_T, self.f_2)
        elif self.model == 2:
            self.f_2 = 1
        elif self.model == 7:
            pass
        elif self.model == 8:
            pass

        self.theta_1 = Constants.theta_1

        # H_deltaT != 0 for M8
        if self.model == 8:
            pass
            # Initial mdot value???
        else:
            self.H_deltaT = Constants.H_deltaT

        self.D_0 = D
        self.D = np.array(D)
        m = (self.rho_d*np.pi*self.D**(3))/(6)  # initial mass
        self.m_d0 = m
        self.m_d = np.array(m)

        self.tau_d0 = self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)
        self.tau_h = self.tau_d0 * ((3*self.Pr_G)/(2*self.f_2*self.theta_1))
        self.u_s = np.linalg.norm(self.get_vel_fluid(self)-self.vel)

        self.Sh = 2 + 0.552*self.Re_d**(0.5)*self.Sc_G**(1/3)
        self.Nu = 2 + 0.552*self.Re_d**(0.5)*self.Pr_G**(1/3)
        self.ODE_solver = ODE_solver

        self.pos_history = []
        self.vel_history = []

        self.temp_history = []
        self.temp_history_nd = []
        self.temp_history_nd1 = []
        self.mass_history = []
        self.mass_history_nd = []
        self.diameter_2_history = []
        self.diameter_2_history_nd = []
        self.ss_temp = []
        self.times = []
        self.times_temp_nd = []
        self.times_mass_nd = []

        # print(self.tau_d0)

    def iterate(self, delta_t, implicit=True):
        self.delta_t = delta_t
        self.time += delta_t
        if self.model == 2:
            # print("Time =", self.time)
            self.M2_avg_properties()
        self.mass_fractions()

        # update f2 HT if not the first timestep, M2's f2 update at if below
        if self.time != delta_t:
            self.f2_HT_update()

#        self.get_non_d_numbers()
#        self.u_s = np.linalg.norm(self.get_vel_fluid(self)-self.vel)

#        self.iterate_velocity(delta_t, implicit)
#        self.iterate_position(delta_t)

        # Insert f2 for M2 here, using original method
        if self.model == 2:

            self.Nu = self.Nu_star
            self.Sh = self.Sh_star
            self.Pr_G = self.Pr_G_bar
            self.Sc_G = self.Sc_G_bar

            self.iterate_mass(delta_t)
            self.m_d_dot = self.mass_ODE(0, self.m_d)

            self.f_2 = ((-self.m_d_dot/(self.m_d*self.B_T_pi)) *
                        (3*self.Pr_G_bar*self.tau_d/self.Nu_star))

            self.iterate_temp(delta_t)
            self.T_d = self.next_temp
            self.m_d = self.next_mass
            # print(self.m_d, self.T_d)

        elif self.coupled == 0:
            self.iterate_mass(delta_t)
            self.m_d_dot = self.mass_ODE(0, self.m_d)
            self.m_d = self.next_mass

        elif self.coupled == 1:
            self.iterate_temp(delta_t)
            self.T_d = self.next_temp

        elif self.coupled == 2:
            self.iterate_mass(delta_t)
            self.m_d_dot = self.mass_ODE(0, self.m_d)
            self.iterate_temp(delta_t)
            self.T_d = self.next_temp
            self.m_d = self.next_mass

#        self.vel = self.next_vel
#        self.pos = self.next_pos
        self.record_state()

    def iterate_velocity(self, delta_t, implicit):
        self.next_vel = self.vel + delta_t * self.get_accel(delta_t, implicit)

    def iterate_position(self, delta_t):
        self.next_pos = self.pos + delta_t * (self.next_vel + self.vel) / 2

    def get_accel(self, delta_t, implicit):
        if not implicit:
            return np.array(self.get_drag_accel() + self.get_gravity(self))
        else:
            return self.get_accel_implicit_drag(delta_t)

    def get_accel_implicit_drag(self, delta_t):
        non_drag_a = self.get_gravity(self)
        v = self.get_vel_fluid(self)
        return((v - self.vel + self.tau_d * non_drag_a) /
               (self.tau_d + delta_t))

    def get_drag_accel(self):
        return(-(self.vel - np.array(self.get_vel_fluid(self))) / self.tau_d)

    def get_tau(self):
        return(self.rho_d * self.D ** 2 / (18 * self.mu_G))

    def get_non_d_numbers(self):
        self.Re_d = (self.rho_G*self.u_s)/(self.mu_G)
        self.Sh = 2 + 0.552*self.Re_d**(0.5)*self.Sc_G**(1/3)
        self.Nu = 2 + 0.552*self.Re_d**(0.5)*self.Pr_G**(1/3)
        return(self.Sh, self.Nu)

    def get_speed(self):
        return vect.mag(self.vel)

    def get_speed_at_time(self, time):
        try:
            index = self.times.index(time)
            return vect.mag(self.vel_history[index])
        except ValueError:
            return 0

    def get_speed_at_index(self, index):
        return vect.mag(self.vel_history[index])

    def get_mass(self):
        return self.density * np.pi * self.diameter ** 3 / 6

    def get_kinetic_energy(self):
        return 0.5 * self.get_mass() + self.get_speed() ** 2

    def iterate_temp(self, delta_t):
        if self.ODE_solver == 0:
            self.next_temp = self.temp_analytic(0, self.T_d, delta_t)

        elif self.ODE_solver == 1:
            self.next_temp = self.forward_euler(
                self.temp_ODE, 0, self.T_d, delta_t)

        elif self.ODE_solver == 2:
            self.next_temp = self.backward_euler_temp(0, self.T_d, delta_t)

        elif self.ODE_solver == 3:
            self.next_temp = self.modified_euler(
                self.temp_ODE, 0, self.T_d, delta_t)

        elif self.ODE_solver == 4:
            self.next_temp = self.runge_kutta(
                self.temp_ODE, 0, self.T_d, delta_t)

        else:
            self.next_temp = self.T_d + delta_t*(
                ((self.f_2*self.Nu) / (3*self.Pr_G)) *
                (self.theta_1/self.tau_d) * (self.T_G-self.T_d))

    def iterate_mass(self, delta_t):
        self.D = ((6*self.m_d)/(self.rho_d*np.pi))**(1/3)
        self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)

        if self.ODE_solver == 0:
            self.next_mass = self.mass_analytic(0, self.m_d, delta_t)

        elif self.ODE_solver == 1:
            self.next_mass = self.forward_euler(
                self.mass_ODE, 0, self.m_d, delta_t)

        elif self.ODE_solver == 2:
            self.next_mass = self.backward_euler_mass(0, self.m_d, delta_t)

        elif self.ODE_solver == 3:
            self.next_mass = self.modified_euler_mass(
                self.mass_ODE, 0, self.m_d, delta_t)

        elif self.ODE_solver == 4:
            self.next_mass = self.runge_kutta_mass(
                self.mass_ODE, 0, self.m_d, delta_t)

        else:
            self.next_mass = self.m_d - delta_t*((((self.Sh) / (3*self.Sc_G)) *
                                                  (self.m_d/self.tau_d)) *
                                                 self.H_M)

    def forward_euler(self, func, t_n, y_n, delta_t):
        return(y_n + delta_t*func(t_n, y_n))

    def backward_euler_mass(self, t_n, m_d, delta_t):
        if self.coupled == 2:
            return((m_d)/(1+delta_t*(((self.Sh)/(3*self.Sc_G)) *
                          (self.H_M/self.tau_d))))
        else:
            return((m_d)/(1+delta_t*(((self.Sh)/(3*self.Sc_G)) *
                          (self.H_M/self.tau_d))))

    def backward_euler_temp(self, t_n, T_d, delta_t):
        if self.coupled == 2:
            return((T_d + delta_t*((((self.f_2*self.Nu)/(3*self.Pr_G)) *
                   (self.theta_1/self.tau_d)*(self.T_G)) +
                ((self.L_V*self.m_d_dot)/(self.C_L*self.m_d))-self.H_deltaT)) /
                (1+delta_t*(((self.f_2*self.Nu)/(3*self.Pr_G)) *
                            (self.theta_1/self.tau_d))))

        else:
            return((T_d + delta_t*((((self.f_2*self.Nu)/(3*self.Pr_G)) *
                   (self.theta_1/self.tau_d)*(self.T_G)))) /
                   (1+delta_t*(((self.f_2*self.Nu)/(3*self.Pr_G)) *
                               (self.theta_1/self.tau_d))))

    def modified_euler(self, func, t_n, y_n, delta_t):
        p_n = self.forward_euler(func, t_n, y_n, delta_t)
        t_n1 = t_n + delta_t
        return(y_n + 0.5*delta_t*(func(t_n, y_n) + func(t_n1, p_n)))

    def runge_kutta(self, func, t_n, y_n, delta_t):
        k_1 = delta_t*func(t_n, y_n)
        k_2 = delta_t*func(t_n + delta_t/2, y_n + 0.5*k_1)
        k_3 = delta_t*func(t_n + delta_t/2, y_n + 0.5*k_2)
        k_4 = delta_t*func(t_n + delta_t, y_n + k_3)
        return(y_n + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4))

    def runge_kutta_mass(self, func, t_n, y_n, delta_t):
        k_1 = delta_t*func(t_n, y_n)
        self.D = ((6*(self.m_d))/(self.rho_d*np.pi))**(1/3)
        self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)
        k_2 = delta_t*func(t_n + delta_t/2, y_n + 0.5*k_1)
        self.D = ((6*(k_2+self.m_d))/(self.rho_d*np.pi))**(1/3)
        self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)
        k_3 = delta_t*func(t_n + delta_t/2, y_n + 0.5*k_2)
        self.D = ((6*(k_3+self.m_d))/(self.rho_d*np.pi))**(1/3)
        self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)
        k_4 = delta_t*func(t_n + delta_t, y_n + k_3)
        return(y_n + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4))

    def modified_euler_mass(self, func, t_n, y_n, delta_t):
        k_1 = delta_t*func(t_n, y_n)
        self.D = ((6*(self.m_d))/(self.rho_d*np.pi))**(1/3)
        self.tau_d = (self.rho_d*self.D**2)/(18*self.mu_G)
        k_2 = delta_t*func(t_n + delta_t/2, y_n + 0.5*k_1)
        return(y_n + 0.5 * (k_1 + k_2))

    def mass_fractions(self):
        # Rearrange if statements, M7/8 requires xsneq, ysneq, bmneq
        self.x_s_eq = (self.P_atm / self.P_G) * np.exp(
            ((self.L_V) / (self.R_bar / self.W_V)) *
            ((1 / self.T_B) - (1 / self.T_d)))
        #print(self.T_d, self.x_s_eq, self.time)
        self.Y_s_eq = ((self.x_s_eq) /
                       (self.x_s_eq + (1 - self.x_s_eq) * self.theta_2))

        self.B_m_eq = (self.Y_s_eq - self.Y_G) / (1 - self.Y_s_eq)

        # different H_M depending on model
        if self.model == 1 or self.model == 7 or self.model == 8:
            self.H_M = np.log(1 + self.B_m_eq)
        elif self.model == 2:  # correction factor
            self.H_M = np.log(1 + self.B_m_eq)
            self.F_M = (((1+self.B_m_eq)**0.7) * (np.log(1+self.B_m_eq)) /
                        self.B_m_eq)
            # Sh_star=Sh when Re=0
            self.Sh_star = 2 + ((self.Sh_0-2) / self.F_M)

            # iterate Nu*, φ, BT'
            def BTpi_iter(BT0):
                tol = 1e-12
                B_T_pi = BT0 + 1
                # Initial values
                F_T = ((((1+BT0)**0.7)/BT0) * np.log(1+BT0))
                Nu_star = 2 + ((self.Nu_0-2) / F_T)
                phi = ((self.C_p_V_bar / self.C_p_g_bar) *
                       (self.Sh_star / Nu_star) * (1 / self.Le_bar))
                B_T_pi = ((1+self.B_m_eq)**phi) - 1
                # print(B_T_pi)

                while abs(B_T_pi - BT0) > tol:
                    BT0 = B_T_pi
                    F_T = ((((1+BT0)**0.7)/BT0) * np.log(1+BT0))
                    Nu_star = 2 + ((self.Nu_0-2) / F_T)
                    phi = ((self.C_p_V_bar / self.C_p_g_bar) *
                           (self.Sh_star / Nu_star) * (1 / self.Le_bar))
                    B_T_pi = ((1+self.B_m_eq)**phi) - 1

                return F_T, Nu_star, phi, B_T_pi

            self.F_T = BTpi_iter(1e-3)[0]
            self.Nu_star = BTpi_iter(1e-3)[1]
            self.phi = BTpi_iter(1e-3)[2]
            self.B_T_pi = BTpi_iter(1e-3)[3]

            # if self.time == self.delta_t: # Initial state
            #     self.Nu_star = self.Nu_0
            # else:
            #     self.Nu_star = 2 + ((self.Nu_0-2) / self.F_T)
            # self.phi = ((self.C_p_V_bar / self.C_p_g_bar) *
            #             (self.Sh_star / self.Nu_star) * (1 / self.Le_bar))
            # self.B_T_pi = ((1+self.B_m_eq)**self.phi)-1
            # # This F_T is for next iteration
            # # print(self.C_p_V_bar, self.C_p_g_bar, self.Sh_star, self.Nu_star, self.Le_bar)
            # self.F_T = ((((1+self.B_T_pi)**0.7)/self.B_T_pi)*
            #             np.log(1+self.B_T_pi))

        elif self.model == 3 or self.model == 4:
            self.H_M = self.B_m_eq
        elif self.model == 5 or self.model == 6:
            self.H_M = self.Y_s_eq - self.Y_G
        elif self.model == 7 or self.model == 8:
            pass

    def f2_HT_update(self):
        # update f2 for M2/4/6/7/8, update H_deltaT for M8
        if self.model == 2:
            pass
        elif self.model == 4 or self.model == 6:
            self.B_T = (self.T_G - self.T_d) * self.C_p_V / self.L_V
            self.f_2 = (1 - self.B_T)**-1
            print(self.T_d, self.B_T, self.f_2)
        elif self.model == 7:
            self.beta = -(3*self.Pr_G*self.tau_d/2) * (self.m_d_dot/self.m_d)
            self.G = self.beta / (np.exp(self.beta)-1)
            self.f_2 = self.G
        """elif self.model==8:
            self.beta = -(3*self.Pr_G*self.tau_d/2) * (self.m_d_dot/self.m_d)
            self.G = self.beta / (np.exp(self.beta)-1)
            self.f_2 = self.G"""

    def M2_avg_properties(self):
        # prerequisites
        self.x_s_eq = (self.P_atm / self.P_G) * np.exp(
            ((self.L_V) / (self.R_bar / self.W_V)) *
            ((1 / self.T_B) - (1 / self.T_d)))
        #print(self.T_d, self.x_s_eq, self.time)
        self.Y_s_eq = ((self.x_s_eq) /
                       (self.x_s_eq + (1 - self.x_s_eq) * self.theta_2))
        self.B_m_eq = (self.Y_s_eq - self.Y_G) / (1 - self.Y_s_eq)
        # Actual calculation
        self.Y_bar = self.Y_s_eq + (1/3) * (self.Y_G - self.Y_s_eq)
        self.T_bar = self.T_G + 1/3 * (self.T_G - self.T_d)
        self.rho_g_bar = ((self.Y_s_eq/self.rho_d)+((1-self.Y_s_eq) /
                                                    self.rho_G)**-1)

        if self.drop_species == "water":
            self.L_V = (2.257e6 + (2.595e3 * (373.15-self.T_d)))
            self.C_p_V_bar = (8137 - 37.34*self.T_bar +
                              (0.07482*(self.T_bar**2)) -
                              (4.956e-5*(self.T_bar**3)))
            self.lambda_bar = (1.024e-2-((8.21e-6)*self.T_bar) +
                               ((1.41e-7)*(self.T_bar**2)) -
                               ((4.51e-11)*(self.T_bar**3)))
            self.G_bar = 1
            self.mu_bar = (4.07e-8 * self.T_bar) - 3.077e-6
            self.C_p_g_bar = self.C_p_V_bar*self.Y_bar + self.C_p_G*(1-self.Y_s_eq)
            self.Le_bar = 1
        if self.drop_species == "decane":
            self.L_V = 3.958e4 * ((619 - self.T_d)**0.38)
            self.T_bar_star = self.T_bar / 1000
            if self.T_bar_star <= 0.8:
                self.C_p_V_bar = (106.6 + 5765*self.T_bar_star -
                                  1675*(self.T_bar_star**2)
                                  + 473.1*(self.T_bar_star**3))
            else:
                self.C_p_V_bar = (411.1 + 5460*self.T_bar_star -
                                  2483*(self.T_bar_star**2)
                                  + 422.9*(self.T_bar_star**3))
            self.lambda_bar = 1.214e-2 * ((self.T_bar/300)**1.8)
            self.G_bar = 5.46e-6 * ((self.T_bar / 300)**1.583)
            self.mu_bar = 5.64e-6 + (1.75e-8*(self.T_bar - 300))
            self.C_p_g_bar = self.C_p_V_bar*self.Y_bar + self.C_p_G*(1-self.Y_s_eq)
            self.Le_bar = (self.lambda_bar /
                           (self.rho_g_bar*self.G_bar*self.C_p_g_bar))
        if self.drop_species == "hexane":
            self.L_V = 5.1478e5 * ((1 - (self.T_d/512))**0.3861)
            self.C_p_V_bar = (-51.31 + 6.767*self.T_bar -
                              (3.626e-3*(self.T_bar**2)))
            self.lambda_bar = (1.112e-2 + ((3.837e-5)*(self.T_bar)) +
                               ((3.778e-8)*(self.T_bar**2)))
            self.G_bar = 1
            self.mu_bar = 5.592e-6 + (5.622e-9 * self.T_bar)
            self.C_p_g_bar = self.C_p_V_bar*self.Y_bar + self.C_p_G*(1-self.Y_s_eq)
            self.Le_bar = 1

        # print(self.mu_bar , self.rho_g_bar, self.C_p_g_bar, self.gamma_bar, self.G_bar)
        self.Pr_G_bar = self.mu_bar * self.C_p_g_bar / self.lambda_bar
        self.Sc_G_bar = self.mu_bar / (self.rho_g_bar * self.G_bar)
        # self.Re_d = PRESET
        self.Sh_0 = 2 + 0.552*(self.Re_d**(0.5))*(self.Sc_G_bar**(1/3))
        self.Nu_0 = 2 + 0.552*(self.Re_d**(0.5))*(self.Pr_G_bar**(1/3))
        # Replace regular parameters with bar parameters

    def mass_ODE(self, t_n, m_d):
        if self.coupled == 2:
            delta_m = -((((self.Sh)/(3*self.Sc_G))*(m_d/self.tau_d))*self.H_M)
        else:
            delta_m = -((((self.Sh)/(3*self.Sc_G))*(m_d/self.tau_d))*self.H_M)
        return(delta_m)

    def temp_ODE(self, t_n, T_d):
        # returns delta T per timestep
        if self.coupled == 2:
            # coupled temp ODE from mass ODE
            delta_T = ((((self.f_2*self.Nu)/(3*self.Pr_G)) *
                       (self.theta_1/self.tau_d)*(self.T_G-T_d)) +
                       ((self.L_V*self.m_d_dot)/(self.C_L*self.m_d)) -
                       self.H_deltaT)
        else:
            # decoupled temp ODE from mass ODE (LV CL delta T ignored)
            delta_T = (((self.f_2*self.Nu)/(3*self.Pr_G)) *
                       (self.theta_1/self.tau_d)*(self.T_G-T_d))
        return(delta_T)

    def temp_analytic(self, t_n, T_d, delta_t):
        # returns T_d_n+1, equation 7.2 in A.Kernan
        return(self.T_G - (self.T_G-T_d)*np.exp(-((self.f_2*self.Nu) /
                                                  (3*self.Pr_G)) *
                                                 (self.theta_1 /
                                                  self.tau_d0)*delta_t))

    def mass_analytic(self, t_n, m_d, delta_t):
        return((m_d**(2/3) - delta_t*(((2*self.Sh*self.mu_G*self.H_M) /
                                      (self.Sc_G*3)*(6/self.rho_d)**(1/3) *
                                      np.pi**(2/3))))**(3/2))

    def record_state(self):
        self.vel_history.append(self.vel.copy())
        self.temp_history.append(self.T_d.copy())
        self.temp_history_nd.append((self.T_d.copy()-self.T_d0) /
                                    (self.T_G-self.T_d0))
        self.temp_history_nd1.append((self.T_d.copy()-self.T_G) /
                                     (self.T_d0-self.T_G))
        self.mass_history.append(self.m_d.copy())
        self.mass_history_nd.append(self.m_d.copy()/self.m_d0)
        self.diameter_2_history.append((self.D.copy()**2)*1000000)
        self.diameter_2_history_nd.append(((self.D.copy()**2)*1000000) /
                                          ((self.D_0**2)*1000000))
        self.times.append(self.time)
        self.times_mass_nd.append(self.time/self.tau_d0)
#        self.ss_temp.append(self.T_G-((-self.H_deltaT+((self.L_V*self.m_d_dot)/(self.C_L*self.m_d)))/((self.f_2*self.Nu)/(3*self.Pr_G))*(self.theta_1/self.tau_d)))
        if self.coupled == 2:
            self.times_temp_nd.append(self.time/self.tau_d)
        else:
            self.times_temp_nd.append(self.time/self.tau_h)
