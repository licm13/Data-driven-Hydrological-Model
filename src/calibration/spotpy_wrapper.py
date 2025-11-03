"""
SPOTPY校准框架包装器
用于统一校准过程驱动模型
"""
import sys
import os
import numpy as np
import spotpy
from spotpy.parameter import Uniform
from typing import Dict, Callable, List

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from metrics.kge import kge
from metrics.entropy import conditional_entropy

class SpotpySetup:
    """
    SPOTPY校准设置类
    
    将水文模型包装为SPOTPY可以校准的格式
    """
    
    def __init__(self,
                 model,
                 precip: np.ndarray,
                 temp: np.ndarray,
                 pet: np.ndarray,
                 discharge_obs: np.ndarray,
                 warmup_period: int = 365,
                 objective_function: str = 'kge'):
        """
        Parameters:
        -----------
        model : BaseHydrologicalModel, 待校准的模型
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        discharge_obs : array, 观测径流
        warmup_period : int, 预热期长度
        objective_function : str, 目标函数 ('kge', 'nse', 'entropy')
        """
        self.model = model
        self.precip = precip
        self.temp = temp
        self.pet = pet
        self.discharge_obs = discharge_obs
        self.warmup_period = warmup_period
        self.objective_function = objective_function
        
        # 获取参数范围
        self.param_bounds = model.get_parameter_bounds()
        self.param_names = list(self.param_bounds.keys())
        
        # 创建SPOTPY参数
        self.params = []
        for param_name in self.param_names:
            low, high = self.param_bounds[param_name]
            self.params.append(
                Uniform(param_name, low=low, high=high)
            )
    
    def parameters(self):
        """返回SPOTPY参数定义"""
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, vector):
        """
        运行模型模拟
        
        Parameters:
        -----------
        vector : array, 参数向量
        
        Returns:
        --------
        discharge_sim : array, 模拟径流
        """
        # 构建参数字典
        params = dict(zip(self.param_names, vector))
        
        # 初始化模型
        self.model.initialize(params)
        
        # 运行模拟
        try:
            discharge_sim = self.model.simulate(
                self.precip, 
                self.temp, 
                self.pet,
                warmup_steps=self.warmup_period
            )
        except Exception as e:
            print(f"Simulation failed: {e}")
            # 返回全0以表示失败
            discharge_sim = np.zeros(len(self.discharge_obs) - self.warmup_period)
        
        return discharge_sim
    
    def evaluation(self):
        """返回观测数据（去除预热期）"""
        return self.discharge_obs[self.warmup_period:]
    
    def objectivefunction(self, simulation, evaluation, params=None):
        """
        计算目标函数值
        
        SPOTPY的约定：返回单个标量，越大越好
        """
        # 移除NaN
        mask = ~(np.isnan(simulation) | np.isnan(evaluation))
        sim = simulation[mask]
        obs = evaluation[mask]
        
        if len(sim) == 0:
            return -9999  # 失败的模拟
        
        if self.objective_function == 'kge':
            return kge(obs, sim)
        elif self.objective_function == 'nse':
            from metrics.kge import nse
            return nse(obs, sim)
        elif self.objective_function == 'entropy':
            # 条件熵：越小越好，所以取负数
            H_cond = conditional_entropy(obs, sim, n_bins=12)
            return -H_cond
        else:
            raise ValueError(f"Unknown objective function: {self.objective_function}")


def calibrate_model(model,
                   precip: np.ndarray,
                   temp: np.ndarray,
                   pet: np.ndarray,
                   discharge_obs: np.ndarray,
                   algorithm: str = 'lhs',
                   n_iterations: int = 1000,
                   warmup_period: int = 365,
                   objective_function: str = 'kge',
                   parallel: str = 'seq',
                   n_jobs: int = 1) -> Dict:
    """
    校准水文模型
    
    Parameters:
    -----------
    model : BaseHydrologicalModel, 待校准模型
    precip, temp, pet : array, 气象输入
    discharge_obs : array, 观测径流
    algorithm : str, 采样算法 ('lhs', 'mc', 'sceua', 'dds')
    n_iterations : int, 采样次数
    warmup_period : int, 预热期
    objective_function : str, 目标函数
    parallel : str, 并行模式
    n_jobs : int, 并行作业数
    
    Returns:
    --------
    results : dict, 包含最优参数和性能
    """
    # 创建SPOTPY设置
    setup = SpotpySetup(
        model=model,
        precip=precip,
        temp=temp,
        pet=pet,
        discharge_obs=discharge_obs,
        warmup_period=warmup_period,
        objective_function=objective_function
    )
    
    # 选择采样算法
    if algorithm.lower() == 'lhs':
        sampler = spotpy.algorithms.lhs(
            setup, 
            dbname='spotpy_lhs', 
            dbformat='ram',
            parallel=parallel,
            save_sim=True
        )
    elif algorithm.lower() == 'mc':
        sampler = spotpy.algorithms.mc(
            setup,
            dbname='spotpy_mc',
            dbformat='ram',
            parallel=parallel,
            save_sim=True
        )
    elif algorithm.lower() == 'sceua':
        sampler = spotpy.algorithms.sceua(
            setup,
            dbname='spotpy_sceua',
            dbformat='ram',
            parallel=parallel,
            save_sim=True
        )
    elif algorithm.lower() == 'dds':
        sampler = spotpy.algorithms.dds(
            setup,
            dbname='spotpy_dds',
            dbformat='ram',
            parallel=parallel,
            save_sim=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 运行采样
    print(f"Calibrating {model.name} using {algorithm.upper()} with {n_iterations} iterations...")
    
    # 根据算法选择采样方法
    if algorithm.lower() in ['lhs', 'mc']:
        # LHS和MC采样器只接受repetitions参数
        sampler.sample(n_iterations)
    else:
        # 其他算法可能支持n_jobs参数
        try:
            sampler.sample(n_iterations, n_jobs=n_jobs)
        except TypeError:
            # 如果不支持n_jobs，则只传递n_iterations
            sampler.sample(n_iterations)
    
    # 获取结果
    results_obj = sampler.getdata()
    
    # 提取最优参数
    best_idx = np.argmax(results_obj['like1'])
    best_params = {}
    for param_name in setup.param_names:
        best_params[param_name] = results_obj['par' + param_name][best_idx]
    
    best_objective = results_obj['like1'][best_idx]
    
    # 用最优参数重新运行
    model.initialize(best_params)
    best_simulation = model.simulate(precip, temp, pet, warmup_steps=warmup_period)
    
    results = {
        'best_params': best_params,
        'best_objective': best_objective,
        'best_simulation': best_simulation,
        'all_results': results_obj,
    }
    
    print(f"Calibration complete. Best {objective_function}: {best_objective:.4f}")
    
    return results