import numpy as np

## ------------ PROPORTIONAL INTEGRAL CONTROLLER ------------ ##
class PI:
    def __init__(self, Kp_d=1, Ki_d=1e3, Kp_q=1, Ki_q=1e3, max_d=200, max_q=200, dt=1/10e3):
        # Controller gains
        self.Kp_d = Kp_d
        self.Ki_d = Ki_d
        self.Kp_q = Kp_q
        self.Ki_q = Ki_q

        # Saturation values 
        self.max_d = max_d
        self.max_q = max_q
        
        # Sampling time
        self.dt = dt

    def reset(self, Id0=0, Iq0=0):
        # Reset initial values for accumulation variables
        self.x_n_d = 0
        self.x_n_q = 0
        self.antiwindup_d = Id0
        self.antiwindup_q = Iq0

    def action_d(self, Idref, Id):
        # Error
        e = Idref-Id

        # ---------- Proportional ---------- #
        P = self.Kp_d*e

        # ---------- Integral ---------- #
        # Forward Euler
        # x(n+1) = x(n) + K*T*u(n)
        # y(n)   = x(n)
        # E.g.,
        # k = 0
        #   y(0) = IC
        #   x(1) = y(0) + K*T*u(0)
        # k = 1
        #   y(1) = x(1)
        #   x(2) = y(1) + K*T*u(1)
        # k = n
        #   y(n) = x(n)
        #   x(n+1) = y(n) + K*T*u(n)
        # K = 1 ; T = Ts
        I    = self.x_n_d
        x_next = I + self.dt*(self.Ki_d*e + self.antiwindup_d)

        # Saturation 
        pre_sat = P + I
        sat     = np.clip(pre_sat,-self.max_d,self.max_d)

        # Calculation of antiwindup for next step
        self.antiwindup_d = (sat-pre_sat)/self.dt

        # Update x_n for next iteration
        self.x_n_d = x_next

        # Return saturated value
        return sat
    
    def action_q(self,Iqref,Iq):
        # Error
        e = Iqref-Iq

        # ---------- Proportional ---------- #
        P = self.Kp_q*e

        # ---------- Integral ---------- #
        # Forward Euler
        # x(n+1) = x(n) + K*T*u(n)
        # y(n)   = x(n)
        # E.g.,
        # k = 0
        #   y(0) = IC
        #   x(1) = y(0) + K*T*u(0)
        # k = 1
        #   y(1) = x(1)
        #   x(2) = y(1) + K*T*u(1)
        # k = n
        #   y(n) = x(n)
        #   x(n+1) = y(n) + K*T*u(n)
        # K = 1 ; T = Ts
        I    = self.x_n_q
        x_next = I + self.dt*(self.Ki_q*e + self.antiwindup_q)

        # Saturation 
        pre_sat = P + I
        sat     = np.clip(pre_sat,-self.max_q,self.max_q)

        # Calculation of antiwindup for next step
        self.antiwindup_q = (sat-pre_sat)/self.dt

        # Update x_n for next iteration
        self.x_n_q = x_next

        # Return saturated value
        return sat