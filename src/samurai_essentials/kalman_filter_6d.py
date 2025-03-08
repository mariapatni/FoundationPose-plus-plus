import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class KalmanFilter6D(object):
    """
    6D Kalman filter for tracking 6DOF poses.
    
    State space (12-dimensional):
    [tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]
    """
    
    def __init__(self):
        ndim, dt = 6, 1.0  # 6 pose dimensions
        
        # State transition matrix (12x12)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Observation matrix (6x12)
        self._update_mat = np.eye(ndim, 2 * ndim)    # 6*12
        self._update_mat_xy = np.zeros((2, 2*ndim))  # 2*12
        # Diagonal
        self._update_mat_xy[0, 0] = 1
        self._update_mat_xy[1, 1] = 1

        # 新增xy传感器噪声参数
        self._std_weight_trans_xy = 1./40  # xy传感器噪声相对小
        # Process noise parameters
        self._std_weight_trans = 1./10    # Translation uncertainty
        self._std_weight_rot = 1./20     # Rotation uncertainty
        self._std_weight_vel_trans = 1./20     # Translation speed uncertainty
        self._std_weight_vel_rot = 1./40    # Rotation speed uncertainty

    def initiate(self, measurement):
        # 用于在最开始的时候返回一个均值和方差
        """Initialize track with unassociated measurement (6DOF pose)"""
        mean_pos = measurement  # mean代表平均值，是期望值，先令期望值等于初始测量值
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        # Initialize uncertainties
        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)    # 计算L2范数（平方根均值），得到空间尺度
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)
        
        # 初始误差比较大
        std = [
            # Position uncertainties
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            
            # Velocity uncertainties
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        scale_xyz = mean[2]     
        scale_rot = mean[5]     
        
        # 预测状态的不确定性，参照samurai，选取一个维度，这里选取z，因为x y 要进一步参与更新
        std_pos = [
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
        ]
        
        std_vel = [
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        # 更新状态
        mean = np.dot(mean, self._motion_mat.T)
        # 将估计过程的方差累积上去
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance

    def project(self, mean, covariance):
        """Project state to measurement space"""
        # Initialize uncertainties
        # 测量误差，参照samurai也用均值
        scale_xyz = mean[2]     
        scale_rot = mean[5]     
        
        # 只考虑可以测量的部分的误差
        std = [
            0.1 * self._std_weight_trans * scale_xyz,
            0.1 * self._std_weight_trans * scale_xyz,
            0.1 * self._std_weight_trans * scale_xyz,
            0.1 * self._std_weight_rot * scale_rot,
            0.1 * self._std_weight_rot * scale_rot,
            0.1 * self._std_weight_rot * scale_rot,
        ]
        innovation_cov = np.diag(np.square(std))
        
        # Project state
        # 均值提取前几维，方差继续传播
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        
        return projected_mean, projected_cov
    
    def project_for_xy(self, mean, covariance):
        # 投影到 x y测量的误差
        scale_xy = max(np.linalg.norm(mean[:2]), 1e-5)
        
        # xy传感器噪声模型
        std_xy = [
            self._std_weight_trans_xy * scale_xy,
            self._std_weight_trans_xy * scale_xy
        ]
        innovation_cov = np.diag(np.square(std_xy))
        
        projected_mean = np.dot(self._update_mat_xy, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat_xy, covariance, self._update_mat_xy.T)) + innovation_cov
        
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # 通过误差计算卡尔曼增益
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        
        # Update with measurement
        innovation = measurement - projected_mean   # 实际测量值-投影预测值，来产生修正
        # 增益加权
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # 修正后方差减少了
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
    
    def update_from_xy(self, mean, covariance, measurement_xy):
        """Kalman filter correction step for xy"""
        proj_mean, proj_cov = self.project_for_xy(mean, covariance)
        
        # 计算卡尔曼增益
        chol_factor, lower = scipy.linalg.cho_factor(
            proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat_xy.T).T,
            check_finite=False).T
        
        # Update with measurement
        innovation = measurement_xy - proj_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, proj_cov, kalman_gain.T))
        
        return new_mean, new_covariance

    # 高效率的多次Predict
    # def multi_predict(self, mean_arr, covariance_arr):
    #     """Vectorized prediction for multiple tracks"""
    #     N = mean_arr.shape[0]
    #     trans = mean_arr[:, :3]
    #     scales = np.linalg.norm(trans, axis=1)
    #     scales[scales == 0] = 1e-5
        
    #     # Build process noise covariance
    #     std_pos = [
    #         self._std_weight_trans * scales,
    #         self._std_weight_trans * scales,
    #         self._std_weight_trans * scales,
    #         1e-2 * np.ones(N),
    #         1e-2 * np.ones(N),
    #         1e-2 * np.ones(N)
    #     ]
    #     std_vel = [
    #         self._std_weight_vel_trans * scales,
    #         self._std_weight_vel_trans * scales,
    #         self._std_weight_vel_trans * scales,
    #         1e-5 * np.ones(N),
    #         1e-5 * np.ones(N),
    #         1e-5 * np.ones(N)
    #     ]
    #     sqr = np.square(np.vstack(std_pos + std_vel)).T
    #     motion_cov = np.array([np.diag(v) for v in sqr])
        
    #     # Predict state
    #     mean_arr = np.dot(mean_arr, self._motion_mat.T)
    #     covariance_arr = np.matmul(
    #         np.matmul(self._motion_mat, covariance_arr), 
    #         self._motion_mat.T.transpose(0,2,1)) + motion_cov
        
    #     return mean_arr, covariance_arr
    

    # 距离判断（用来判断是否可能相差太远）
    # def gating_distance(self, mean, covariance, measurements,
    #                     only_position=False, metric='maha'):
    #     """Compute gating distance between state and measurements"""
    #     projected_mean, projected_cov = self.project(mean, covariance)
        
    #     if only_position:
    #         # Use only translation components
    #         projected_mean = projected_mean[:3]
    #         projected_cov = projected_cov[:3, :3]
    #         measurements = measurements[:, :3]
        
    #     d = measurements - projected_mean
        
    #     if metric == 'maha':
    #         cholesky_factor = np.linalg.cholesky(projected_cov)
    #         z = scipy.linalg.solve_triangular(
    #             cholesky_factor, d.T, lower=True, 
    #             check_finite=False, overwrite_b=True)
    #         return np.sum(z * z, axis=0)
    #     else:
    #         return np.sum(d * d, axis=1)