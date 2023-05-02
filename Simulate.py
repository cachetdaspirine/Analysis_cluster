import numpy as np
from multiprocessing import Pool
import sys
import math
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Aging_Condensate/Gillespie_backend')
import Gillespie_backend as Gil
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

class ParallelSimulation:
    """
    This class is used to quickly simulate a system, extract some parameters on several nodes.
    To make a simulation, one need to edit a child class and write the extract_parameter function

    gillespie_param is a dictionnary. One or several variable is an array, a list of parameters compatible with the 
    use of a starmap is then created.
                
    """
    def __init__(self,
                 gillespie_param,
                 simulation_param,
                 cpu_param):
        self.gillespie_param = gillespie_param
        self.step_tot = simulation_param['step_tot'] # total number of steps
        self.dump_step = simulation_param['dump_step'] # number of steps between two outputs
        self.min_dump_step = simulation_param['min_dump_step'] # minimal number of steps before a measurement is performed
        self.measurement_step = simulation_param['measurement_steps'] # number of steps use to compute a thermodynamic quantity
        self.Nnodes = cpu_param['Nnodes']
        self.Simulation_Name = simulation_param['Simulation_Name']
        try:
            self.keyarray = simulation_param['label_key']
        except KeyError:
            self.keyarray = False
        self.gillespieParamNames = ['ell_tot','rho0','BindingEnergy','kdiff','seed','sliding','Nlinker','old_gillespie','dimension']
        self.default_values = {'ell_tot':100,'rho0':0.1,'BindingEnergy':-1.,'kdiff':1.,'seed':19874,'sliding':False,'Nlinker':0,'old_gillespie':None,'dimension':3}
        self.make_args()
    def make_args(self):
        # first find which key has an array as a value:
        ValueLength=0
        for key,value in self.gillespie_param.items():
            if type(value)==np.ndarray or type(value)==list :
                ValueLength = value.__len__()
                if not self.keyarray:
                    self.keyarray = key
        self.args = []
        k = 0
        for key in self.gillespieParamNames:
            if key==self.keyarray:
                self.index_of_keyarray = k
            else:
                k+=1
            try:
                value = self.gillespie_param[key]
            except KeyError:
                value = self.default_values[key]
            if type(value)!=np.ndarray and type(value)!=list:
                self.args.append([value for _ in range(ValueLength)])
            else :
                self.args.append(value)
        self.args = np.transpose(self.args)
        self.args = [tuple(row) for row in self.args]
    def Run_One_system(self,a1,a2,a3,a4,a5,a6,a7,a8,a9):
        arg = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
        res = dict()
        gillespie  = Gil.Gillespie(a1,a2,a3,a4,a5,a6,a7,a8,a9)
        for step in range(self.step_tot//self.dump_step):
            if step*self.dump_step >= self.min_dump_step:
                move,time = gillespie.evolve(self.dump_step-self.measurement_step)
                for measuring_steps in range(self.measurement_step):
                    self.state_b4(gillespie,time)
                    mes_move,mes_time = gillespie.evolve()
                    move =  np.append(move,mes_move)
                    time = np.append(time,mes_time)
                    self.state_after(gillespie,time)
                res.update(self.extract_parameter(gillespie,move,time,(step+1)*self.dump_step,name = str(truncate(arg[self.index_of_keyarray],3))))
            else :
                move,time = gillespie.evolve(self.dump_step)
        print(str(truncate(arg[self.index_of_keyarray],3))+" is over.")
        return res
    def Parallel_Run(self):
        pool = Pool(self.Nnodes)
        res = pool.starmap(self.Run_One_system,self.args)
        return self.unpack_res(res)
    def state_b4(self,gillespie,time):
        """ this function save what has to be saves before an evolution step"""
    def state_after(self,gillespie,time):
        """ this function save what has to be saved after an evolution step"""
    def unpack_res(self,res):
        """
        This function unpack the results and has to be edited in the child function
        """
        return
    def extract_parameter(self,gillespie,moves,time,name):
        """
        This function has to be edited in the child class
        name is a way to recognize from which gillespie object the parameters are being extracted.
        """
        return
         
