import copy
import subprocess

from scipy.optimize import fmin_l_bfgs_b
import numpy as np


class MonotonousOptimizer:
    def __init__(self, gt_images, shdom_f_g, z0, reff_power=2/3, lwc_power=1):
        self.gt_measurements = gt_images
        self.shdom_forward_and_grad = shdom_f_g
        self.z0 = z0
        self.reff_pow = reff_power
        self.lwc_pow = lwc_power

    def objective_func(self, x):
        loss = 0
        ret_imgs, grad = self.shdom_forward_and_grad(x)
        for ret_img, gt_img in zip(ret_imgs, self.gt_measurements):
            loss += np.linalg.norm(ret_img-gt_img)**2

        lwc_grad = (z-self.z0)**self.lwc_pow
        reff_grad = (z-self.z0)**self.reff_pow
        grad[0] *= lwc_grad
        grad[1] *= reff_grad

        return loss, grad

    def optimize_monotonous(self, init_reff, init_lwc):
        result = fmin_l_bfgs_b(func=self.objective_func,
                               x0=np.array([init_reff, init_lwc]))

        return result


def optimize_monotonous_init_vals(gt_measurements, shdom_func, z0, init_reff=10, init_lwc=0.1):
    optimizer = MonotonousOptimizer(gt_images=gt_measurements, shdom_f_g=shdom_func, z0=z0)
    res = optimizer.optimize_monotonous(init_reff=init_reff, init_lwc=init_lwc)
    return res


