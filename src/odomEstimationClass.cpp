// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"

void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution)
{
    //init local map
    laserCloudCornerMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudSurfMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

    //downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution, map_resolution, map_resolution);
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);

    //kd-tree
    kdtreeEdgeMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdtreeSurfMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    optimization_count = 2;
}

/**
 * @brief 把系统第一帧的LiDAR角点和面点注册到地图中，完成初始化
 */
void OdomEstimationClass::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_in)
{
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    optimization_count = 12;
}

/**
 * @brief 重点：输入当前帧的点云，对它进行scan-to-map配准，得到当前帧的里程计
 */
void OdomEstimationClass::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_in)
{
    if (optimization_count > 2)
        optimization_count--;

    // Step 1: 利用匀速运动模型对当前帧位姿进行预测
    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();
    
    // Step 2: 对当前帧输出的角点和面点进行降采样，降低计算量
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZI>());
    downSamplingToMap(edge_in, downsampledEdgeCloud, surf_in, downsampledSurfCloud);
    //ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
    
    // Step 3: 如果地图中的角点和面点个数足够，则执行当前帧到地图的配准，并使用ceres优化
    if (laserCloudCornerMap->points.size() > 10 && laserCloudSurfMap->points.size() > 50)
    {
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);

        for (int iterCount = 0; iterCount < optimization_count; iterCount++)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);

            problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

            addEdgeCostFactor(downsampledEdgeCloud, laserCloudCornerMap, 
                problem, loss_function);
            addSurfCostFactor(downsampledSurfCloud, laserCloudSurfMap, 
                problem, loss_function);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);
        }
    }
    else
    {
        printf("not enough points in map to associate, map error");
    }

    // Step 4: 把当前帧的点云根据位姿转到world系下，并注册到地图中，同时更新局部地图
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    addPointsToMap(downsampledEdgeCloud, downsampledSurfCloud);
}

/**
 * @brief 把当前帧的一个点，根据位姿转到world系下
 */
void OdomEstimationClass::pointAssociateToMap(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}

/**
 * @brief 对当前帧的角点和面点降采样，以降低计算量
 */
void OdomEstimationClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_pc_out, const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc_out)
{
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);
}

/**
 * @brief 添加当前帧角点 到 地图角点 的残差
 * 
 * @param[in] pc_in   当前帧角点  
 * @param[in] map_in  地图角点
 * @param[in] problem 
 * @param[in] loss_function 
 */
void OdomEstimationClass::addEdgeCostFactor(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &map_in, 
    ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int corner_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        //; 先把这个角点转到地图下
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        //; 从地图中寻找和这个角点距离最近的5个角点
        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        //; 如果这些地图角点 离 当前帧的角点 的距离足够近，则这个匹配有效
        if (pointSearchSqDis[4] < 1.0)
        {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                    map_in->points[pointSearchInd[j]].y,
                                    map_in->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
            }
            center = center / 5.0;  //; 5个地图角点中心

            //; 5个地图角点的协方差矩阵
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            //; 对角点的协方差矩阵进行分解，得到直线的法向量
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            //; 如果最大特征值 比 第二大特征值 大3倍，说明当前拟合的直线符合要求，则构造点到线的残差
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                //; 点到线的残差块，解析雅克比
                ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(
                    curr_point, point_a, point_b);
                problem.AddResidualBlock(cost_function, loss_function, parameters);
                corner_num++;
            }
        }
    }
    if (corner_num < 20)
    {
        printf("not enough correct points");
    }
}

/**
 * @brief 添加当前帧面点 到 地图面点 的残差
 * 
 * @param[in] pc_in   当前帧面点  
 * @param[in] map_in  地图面点
 * @param[in] problem 
 * @param[in] loss_function 
 */
void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int surf_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        //; 当前帧面点转到world系下，并从地图中寻找距离最近的5个面点
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            //; 对拟合的局部平面分解法向量
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            //; 再依次判断这5个点到拟合的平面的距离是否满足阈值要求，如果距离太大则拟合的平面噪声较大，
            //; 就不使用这个平面
            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            if (planeValid)
            {
                //; 构造点到面的残差块
                ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(
                    curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, parameters);
                surf_num++;
            }
        }
    }
    if (surf_num < 20)
    {
        printf("not enough correct points");
    }
}

/**
 * @brief 把当前帧的角点和面点注册到地图中，并根据当前帧的位置对局部地图的点云进行更新
 */
void OdomEstimationClass::addPointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZI>::Ptr &downsampledSurfCloud)
{
    // Step 1: 把当前帧的点云根据当前帧的位姿，注册到地图中
    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp);
    }
    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
    }

    // Step 2: 根据当前帧的位置，选择它周围100米范围内的点云作为局部地图，用于下一帧的scan-to-map配准
    double x_min = +odom.translation().x() - 100;
    double y_min = +odom.translation().y() - 100;
    double z_min = +odom.translation().z() - 100;
    double x_max = +odom.translation().x() + 100;
    double y_max = +odom.translation().y() + 100;
    double z_max = +odom.translation().z() + 100;

    //ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCorner(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZI>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);

    //; 最后对地图再进行一个降采样
    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
    downSizeFilterEdge.setInputCloud(tmpCorner);
    downSizeFilterEdge.filter(*laserCloudCornerMap);
}

void OdomEstimationClass::getMap(pcl::PointCloud<pcl::PointXYZI>::Ptr &laserCloudMap)
{
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass()
{
}
