'''
MIT License

Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

##def gpu_knn_search(pcl_tensor, k=20):
##    distances = torch.cdist(pcl_tensor, pcl_tensor)
##    distances, indices = torch.topk(distances, k=k, largest=False)
##    return distances, indices
def gpu_knn_search(pcl_tensor, k=10):
    """
    GPU上的K最近邻搜索
    Args:
        pcl_tensor: [N, 3] 点云张量
        k: 最近邻数量（包括自身）
    Returns:
        distances: [N, k] 距离矩阵
        indices: [N, k] 索引矩阵
    """
    N = pcl_tensor.shape[0]
    
    # 特殊情况处理
    if N == 0:
        return torch.empty((0, 0), device=pcl_tensor.device), torch.empty((0, 0), device=pcl_tensor.device, dtype=torch.long)
    elif N == 1:
        # 只有一个点，返回自身
        return torch.zeros((1, 1), device=pcl_tensor.device), torch.zeros((1, 1), device=pcl_tensor.device, dtype=torch.long)
    
    # 计算距离矩阵
    distances = torch.cdist(pcl_tensor, pcl_tensor)
    
    # 调整k值：k不能大于N，且当k=1时我们需要特殊处理
    k = min(k, N)
    
    if k <= 0:
        return torch.empty((N, 0), device=pcl_tensor.device), torch.empty((N, 0), device=pcl_tensor.device, dtype=torch.long)
    
    # 使用安全的topk
    try:
        # 获取k个最近邻（包括自身）
        distances, indices = torch.topk(distances, k=k, largest=False)
        
        # 当N很小时，确保返回正确的形状
        if distances.shape[1] < k:
            # 填充到k的大小
            pad_size = k - distances.shape[1]
            distances = torch.nn.functional.pad(distances, (0, pad_size), value=float('inf'))
            indices = torch.nn.functional.pad(indices, (0, pad_size), value=0)
        
        return distances, indices
    except RuntimeError as e:
        print(f"KNN搜索错误: {e}, N={N}, k={k}")
        # 返回安全的结果
        if N > 0 and k > 0:
            # 返回距离矩阵的前k列
            return distances[:, :k], torch.arange(N, device=pcl_tensor.device).unsqueeze(1).repeat(1, k)
        else:
            return torch.empty((N, 0), device=pcl_tensor.device), torch.empty((N, 0), device=pcl_tensor.device, dtype=torch.long)

def calculate_curvatures_gpu(pcl_tensor, indices):
    """
    计算点云中每个点的曲率
    """
    try:
        neighbors = pcl_tensor[indices]  
        mean = neighbors.mean(dim=1, keepdim=True)  
        centered = neighbors - mean
        
        # 检查邻居数量
        if neighbors.shape[1] < 2:
            # 如果邻居数量不足，返回零曲率
            return torch.zeros(pcl_tensor.shape[0], device=pcl_tensor.device)
        
        covariance = torch.einsum('bij,bik->bjk', centered, centered) / (neighbors.shape[1] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance)  
        
        # 避免除以零
        denominator = eigenvalues.sum(dim=1) + 1e-6
        curvatures = eigenvalues[:, 0] / denominator
        
        # 确保曲率在合理范围内
        curvatures = torch.clamp(curvatures, 0.0, 1.0)
        
        return curvatures
    except Exception as e:
        print(f"曲率计算错误: {e}")
        # 返回零曲率
        return torch.zeros(pcl_tensor.shape[0], device=pcl_tensor.device)

def calculate_gradients_gpu(pcl_tensor, indices):
    """
    计算点云中每个点的梯度
    """
    try:
        neighbors = pcl_tensor[indices]
        
        # 检查邻居数量
        if neighbors.shape[1] < 1:
            return torch.zeros(pcl_tensor.shape[0], device=pcl_tensor.device)
        
        dz = neighbors[:, :, 2] - pcl_tensor[:, 2].unsqueeze(1)  
        gradients = torch.mean(torch.abs(dz), dim=1)
        
        # 确保梯度非负
        gradients = torch.clamp(gradients, min=0.0)
        
        return gradients
    except Exception as e:
        print(f"梯度计算错误: {e}")
        return torch.zeros(pcl_tensor.shape[0], device=pcl_tensor.device)

def extract_features_with_classification_gpu(pcl_arr, curvature_threshold=0.1, gradient_threshold=0.05, voxel_size=0.2, target_num_points=10000):
    """
    使用GPU进行点云特征提取和分类
    
    Args:
        pcl_arr: [N, 4] 点云数组，前三列是XYZ坐标，第四列是强度
        curvature_threshold: 曲率阈值
        gradient_threshold: 梯度阈值
        voxel_size: 下采样的体素大小
        target_num_points: 目标点云数量
    
    Returns:
        processed_pcl: 处理后的点云
    """
    # 检查输入有效性
    if pcl_arr is None or len(pcl_arr) == 0:
        print("警告: 输入点云为空")
        return np.empty((0, 6))
    
    if len(pcl_arr.shape) != 2 or pcl_arr.shape[1] < 4:
        print(f"警告: 点云形状不正确: {pcl_arr.shape}")
        return np.empty((0, 6))
    
    N = len(pcl_arr)
    print(f"处理点云: {N} 个点")
    
    # 如果点云数量太少，直接返回
    if N < 2:
        print("警告: 点云数量太少，无法进行特征提取")
        # 返回格式一致的结果
        if N == 1:
            single_point = np.hstack((pcl_arr[0:1], np.array([[0.0, 0.0]])))
            return single_point
        return np.empty((0, 6))
    
    pcl_positions = pcl_arr[:, :3]  
    pcl_intensities = pcl_arr[:, 3]  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcl_tensor = torch.tensor(pcl_positions, dtype=torch.float32).to(device)

    # 根据点云数量动态调整k值
    k = min(10, max(2, N))  # k至少为2，最大为10
    
    try:
        _, indices = gpu_knn_search(pcl_tensor, k=k)
        
        # 检查indices的有效性
        if indices.shape[1] < 2:
            print(f"警告: KNN返回的邻居数量不足: {indices.shape}")
            # 返回原始点云的前6列
            if pcl_arr.shape[1] >= 6:
                return pcl_arr[:, :6]
            else:
                # 补全到6列
                result = np.zeros((N, 6))
                result[:, :pcl_arr.shape[1]] = pcl_arr
                return result
        
        curvatures = calculate_curvatures_gpu(pcl_tensor, indices)
        gradients = calculate_gradients_gpu(pcl_tensor, indices)
        
        # 将结果移回CPU进行后续处理
        curvatures_cpu = curvatures.cpu().numpy()
        gradients_cpu = gradients.cpu().numpy()
        
        is_feature = (curvatures_cpu > curvature_threshold) | (gradients_cpu > gradient_threshold)
        
        # 提取特征点
        if np.any(is_feature):
            feature_points = pcl_arr[is_feature]
            feature_curvatures = curvatures_cpu[is_feature]
            feature_gradients = gradients_cpu[is_feature]
            feature_points_with_values = np.hstack((
                feature_points, 
                feature_curvatures.reshape(-1, 1), 
                feature_gradients.reshape(-1, 1)
            ))
        else:
            feature_points_with_values = np.empty((0, 6))
        
        # 处理非特征点
        non_feature_mask = ~is_feature
        if np.any(non_feature_mask):
            non_feature_points = pcl_arr[non_feature_mask]
            
            if len(non_feature_points) > 0:
                non_feature_positions = non_feature_points[:, :3]
                
                # 创建Open3D点云
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(non_feature_positions)
                
                # 体素下采样
                downsampled_pcl = point_cloud.voxel_down_sample(voxel_size)
                
                if downsampled_pcl and len(downsampled_pcl.points) > 0:
                    downsampled_positions = np.asarray(downsampled_pcl.points)
                    
                    # 找到下采样点在原始非特征点中的最近邻
                    if len(non_feature_positions) > 0 and len(downsampled_positions) > 0:
                        # 使用更安全的方式查找最近邻
                        try:
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(non_feature_positions)
                            distances, downsampled_indices = nbrs.kneighbors(downsampled_positions)
                            
                            # 获取对应的强度值
                            if len(downsampled_indices) > 0:
                                # 确保索引在有效范围内
                                valid_indices = downsampled_indices.flatten()
                                if np.max(valid_indices) < len(pcl_intensities):
                                    corresponding_intensity_indices = np.where(non_feature_mask)[0][valid_indices]
                                    average_intensities = pcl_intensities[corresponding_intensity_indices].reshape(-1, 1)
                                    
                                    downsampled_points = np.hstack((downsampled_positions, average_intensities))
                                    downsampled_points_extended = np.hstack((downsampled_points, np.zeros((downsampled_points.shape[0], 2))))
                                else:
                                    downsampled_points_extended = np.empty((0, 6))
                            else:
                                downsampled_points_extended = np.empty((0, 6))
                        except Exception as e:
                            print(f"最近邻搜索错误: {e}")
                            downsampled_points_extended = np.empty((0, 6))
                    else:
                        downsampled_points_extended = np.empty((0, 6))
                else:
                    downsampled_points_extended = np.empty((0, 6))
            else:
                downsampled_points_extended = np.empty((0, 6))
        else:
            downsampled_points_extended = np.empty((0, 6))
        
        # 合并结果
        if feature_points_with_values.shape[0] > 0 and downsampled_points_extended.shape[0] > 0:
            processed_pcl = np.vstack((feature_points_with_values, downsampled_points_extended))
        elif feature_points_with_values.shape[0] > 0:
            processed_pcl = feature_points_with_values
        elif downsampled_points_extended.shape[0] > 0:
            processed_pcl = downsampled_points_extended
        else:
            # 如果没有点被处理，返回原始点云的前6列
            if pcl_arr.shape[1] >= 6:
                processed_pcl = pcl_arr[:, :6]
            else:
                processed_pcl = np.hstack((pcl_arr, np.zeros((N, 6 - pcl_arr.shape[1]))))
        
        # 限制点云数量
        if len(processed_pcl) > target_num_points:
            indices = np.random.choice(len(processed_pcl), target_num_points, replace=False)
            processed_pcl = processed_pcl[indices]
        
        print(f"处理完成: {len(processed_pcl)} 个点")
        return processed_pcl
        
    except Exception as e:
        print(f"特征提取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 出错时返回原始点云的前6列
        if pcl_arr.shape[1] >= 6:
            return pcl_arr[:, :6]
        else:
            # 补全到6列
            result = np.zeros((N, 6))
            result[:, :pcl_arr.shape[1]] = pcl_arr
            return result
