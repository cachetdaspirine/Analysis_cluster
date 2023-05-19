import numpy as np
import sys
import math
import matplotlib.pyplot as plt
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend')
import Gillespie_backend as gil
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Analysis_cluster/')
import Simulate as Sim
import os

Ec = lambda L,N : -3/2*np.log(L/N* np.pi/3)

def Get_E_Nlinker_Transition_curve(Gil_args,Npoints=20,step_tot = 10**4):
    """
    Compute the N_linker vs energy transition curve
    Parameters :
    Gil_args (iterable) : parameters of the gillespie simulation in the order : ell_tot,kdiff,Nlinker
    step_tot (int) : total number of steps to equilibrate, the first have of the simulation is to reach a steady state
        the other half of the simulation is for the ensemble average. Usually : no need to change step_tot.
    """
    approx_Ec = Ec(Gil_args[0],Gil_args[2])
    Energy = np.linspace(0.5*approx_Ec,2*approx_Ec,Npoints)
    gillespie_params,simulation_param,cpu_param = parameters_to_dict(Gil_args[0],Energy,Gil_args[1],np.random.randint(0,100000,Npoints),Gil_args[2],step_tot,step_tot//2,step_tot//2,step_tot//2)
    names = [str(truncate(e,3)) for e in Energy]
    steps = [str(step*simulation_param['dump_step']) for step in range(simulation_param['min_dump_step']//simulation_param['dump_step']+1,step_tot//simulation_param['dump_step']+1)]
    Sim = Simulation(gillespie_params,simulation_param,cpu_param,names,steps)
    Res = Sim.Parallel_Run()
    return Res

def parameters_to_dict(ell_tot,Energy,kdiff,seeds,Nlinker,step_tot,equilibration_step,dump_step,measurement_step):
    gillespie_params =  {'ell_tot':ell_tot,'rho0':0.,'BindingEnergy':Energy,'kdiff':kdiff,'seed':seeds,'sliding':False,'Nlinker':Nlinker,'old_gillespie':None,'dimension':3}
    simulation_param = {'step_tot' : step_tot,'min_dump_step':equilibration_step, 'dump_step':dump_step,'label_key':'BindingEnergy','Simulation_Name':'10_linker_L_10E3','measurement_steps':measurement_step}
    cpu_param = {'Nnodes':10}
    return gillespie_params,simulation_param,cpu_param


class Simulation(Sim.ParallelSimulation):
    def __init__(self,gillespie_param,simulation_param,cpu_param,names,steps):
        super().__init__(gillespie_param,simulation_param,cpu_param)
        self.Nlinker=0
        self.time_of_measurement = 0.
        self.names = names
        self.steps=steps
    def extract_parameter(self, gillespie, moves, time,step,name):
        self.Nlinker = self.Nlinker/self.time_of_measurement
        temp_Nlinker = self.Nlinker
        self.Nlinker = 0.
        self.time_of_measurement = 0.
        return {"Nlinker_"+name+"_"+str(step):temp_Nlinker}
    def state_b4(self,gillespie,time):
        """ this function save what has to be saves before an evolution step"""
        self.Nlinker_b4 = gillespie.get_N_loop()-1
    def state_after(self,gillespie,time):
        """ this function save what has to be saved after an evolution step"""
        self.Nlinker += self.Nlinker_b4*(time[-1])
        self.time_of_measurement +=time[-1] # total time of the measurement.
    def unpack_res(self, res):
        concatenate_dict = dict()
        for dictionnary in res:
            concatenate_dict.update(dictionnary)
        Res = []
        for name in self.names:
            for step in self.steps:
                Res.append([float(name),concatenate_dict["Nlinker_"+name+"_"+str(step)]])
        return Res