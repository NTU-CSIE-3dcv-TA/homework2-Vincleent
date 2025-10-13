from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def compute_barycentric_coordinates(pts_3d, Cw):
    homogeneous_Cw = np.hstack([Cw, np.ones((4, 1))])

    alphas = []
    # Solve Pi = Cw @ alpha
    for i in range(pts_3d.shape[0]):
        Pi = np.append(pts_3d[i], 1)
        alpha = np.linalg.inv(homogeneous_Cw.T) @ Pi
        alphas.append(alpha)

    return alphas

def epnp_error(beta, V_ctrl, Cw):
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    Cc = sum(beta[k] * V_ctrl[k] for k in range(len(beta)))
    rho = np.array([np.sum((Cw[a]-Cw[b])**2) for a,b in pairs])
    f = []
    for a,b in pairs:
        diff_c = Cc[a] - Cc[b]
        f.append(np.dot(diff_c, diff_c))
    return np.array(f) - rho  # f(β)

def epnp(pts_3d, pts_2d, cameraMatrix):
    N = pts_3d.shape[0]
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]

    Cw = np.zeros((4, 3))
    centroid = np.mean(pts_3d, axis=0)
    Cw[0] = centroid
    # SVD取代PCA
    U, S, Vt = np.linalg.svd(pts_3d - centroid)
    Cw[1] = centroid + Vt[0] * np.std(pts_3d[:, 0])
    Cw[2] = centroid + Vt[1] * np.std(pts_3d[:, 1])
    Cw[3] = centroid + Vt[2] * np.std(pts_3d[:, 2])

    alphas = compute_barycentric_coordinates(pts_3d, Cw)

    # solve w * u = alphas * cameraMatrix @ Cw
    # 計算2d點投影出去的barycentric 3d點
    M = np.zeros((2*N, 12))
    for i in range(N):
        u, v = pts_2d[i]
        alpha = alphas[i]

        M[2*i] = [alpha[0]*fx, 0, alpha[0]*(cx-u), alpha[1]*fx, 0, alpha[1]*(cx-u), alpha[2]*fx, 0, alpha[2]*(cx-u), alpha[3]*fx, 0, alpha[3]*(cx-u)]
        M[2*i+1] = [0, alpha[0]*fy, alpha[0]*(cy-v), 0, alpha[1]*fy, alpha[1]*(cy-v), 0, alpha[2]*fy, alpha[2]*(cy-v), 0, alpha[3]*fy, alpha[3]*(cy-v)]
    
    _, _, Vt = np.linalg.svd(M)
    tol = 1e-8
    null_mask = S < tol
    N_null = np.sum(null_mask)
    if N_null == 0:
        N_null = 1
    V_null = Vt[-N_null:]  # shape: (N_null, 12)

    # 將每個控制點取出 (每組有3個座標)
    V_ctrl = [v.reshape(4, 3) for v in V_null]

    # 世界座標的距離平方 (6 constraints)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    rho = np.array([np.sum((Cw[a]-Cw[b])**2) for a,b in pairs])

    if N_null == 1:
        v = V_ctrl[0]
        num = 0.0
        den = 0.0
        for a,b in pairs:
            dv = v[a] - v[b]
            dw = Cw[a] - Cw[b]
            nv = np.linalg.norm(dv)
            num += nv * np.linalg.norm(dw)
            den += nv * nv
        beta = [(num / den) if den > 1e-12 else 1.0]

    elif N_null == 2:
        L = np.zeros((6,3))
        for k, (a,b) in enumerate(pairs):
            v1 = V_ctrl[0][a] - V_ctrl[0][b]
            v2 = V_ctrl[1][a] - V_ctrl[1][b]
            L[k] = [np.dot(v1,v1), 2*np.dot(v1,v2), np.dot(v2,v2)]
        beta_lin = np.linalg.pinv(L) @ rho
        b11, b12, b22 = beta_lin
        beta = np.sqrt([abs(b11), abs(b22)])

    elif N_null == 3:
        L = np.zeros((6,6))
        for k,(a,b) in enumerate(pairs):
            v1 = V_ctrl[0][a] - V_ctrl[0][b]
            v2 = V_ctrl[1][a] - V_ctrl[1][b]
            v3 = V_ctrl[2][a] - V_ctrl[2][b]
            L[k] = [
                np.dot(v1,v1), 2*np.dot(v1,v2), 2*np.dot(v1,v3),
                np.dot(v2,v2), 2*np.dot(v2,v3), np.dot(v3,v3)
            ]
        beta_lin = np.linalg.pinv(L) @ rho
        b11, b12, b13, b22, b23, b33 = beta_lin
        beta = np.sqrt([abs(b11), abs(b22), abs(b33)])
    else:
        beta = [1.0]

    # refinement
    res = least_squares(epnp_error, beta, args=(V_ctrl, Cw), method='lm')
    beta = res.x

    if N_null < 4:
        Cc = sum(beta[i] * V_ctrl[i] for i in range(len(beta)))
    else:
        Cc = V_ctrl[-1]

    # Arun's Method
    Pc = (alphas @ Cc)
    Pw = pts_3d
    mu_c = np.mean(Pc, axis=0)
    mu_w = np.mean(Pw, axis=0)

    Xc = Pc - mu_c
    Xw = Pw - mu_w
    H = Xw.T @ Xc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = mu_c - R @ mu_w


    return R, t


def reproj_errors(pts_3d, pts_2d, R, t, cameraMatrix):
    X = (R @ pts_3d.T + t.reshape(3,1)).T
    x_norm = X[:, :2] / X[:, 2:3]
    x_pix_h = (cameraMatrix @ np.hstack([x_norm, np.ones((len(pts_3d),1))]).T).T 
    uv_hat = x_pix_h[:, :2]
    return np.linalg.norm(uv_hat - pts_2d, axis=1)


def pnp_ransac(pts_3d, pts_2d, cameraMatrix, distCoeffs=None, reproj_thresh=4.0, max_iters=2000, min_sample=6, confidence=0.99, seed=42):
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    if distCoeffs is not None:
        pts_2d = cv2.undistortPoints(pts_2d.reshape(-1, 1, 2), cameraMatrix, distCoeffs)
        pts_2d = pts_2d.reshape(-1, 2) * np.array([fx, fy]) + np.array([cx, cy])

    N = pts_3d.shape[0]
    best_inliers = []
    best_R, best_t = None, None

    # 預先打亂索引以加速抽樣
    idx_all = list(range(N))

    # RANSAC 迴圈
    for it in range(max_iters):
        idx = np.random.choice(idx_all, size=min_sample, replace=False)
        R_try, t_try = epnp(pts_3d[idx], pts_2d[idx], cameraMatrix)

        # 2) 數內點
        errs = reproj_errors(pts_3d, pts_2d, R_try, t_try, cameraMatrix)
        inliers = np.where(errs < reproj_thresh)[0]

        # 3) 更新最佳
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R_try, t_try

            # 動態終止條件（依目前內點率估算所需迭代）
            w = len(inliers) / N
            w = np.clip(w, 1e-6, 1-1e-6)
            denom = np.log(1 - (w ** min_sample))
            if denom < 0:
                k_needed = int(np.ceil(np.log(1 - confidence) / denom))
                max_iters = min(max_iters, k_needed)

        # 提早收斂
        if it >= max_iters - 1:
            break

    # 若完全失敗，回傳 None
    if best_R is None:
        return False, None, None, np.array([], dtype=int)

    best_R, _ = cv2.Rodrigues(best_R)
    return True, best_R, best_t, best_inliers





def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    # Hint: you may use "Descriptors Matching and ratio test" first

    # 建立 FLANN 比對器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 進行 knn matching（每個 query descriptor 找兩個 model descriptor）
    matches = flann.knnMatch(desc_query, desc_model, k=2)

    # Ratio test (Lowe’s test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 6:
        return None, None, None, None

    # 取得對應點
    pts_2d = np.float32([kp_query[m.queryIdx] for m in good_matches])
    pts_3d = np.float32([kp_model[m.trainIdx] for m in good_matches])

    retval, rvec, tvec, inliers = pnp_ransac(
        pts_3d, pts_2d,
        cameraMatrix,
        distCoeffs,
        reproj_thresh=8.0,
        confidence=0.99
    )

    return retval, rvec, tvec, inliers


def rotation_error(R1, R2):
    #TODO: calculate rotation error
    w1, x1, y1, z1 = R1[..., 3], R1[..., 0], R1[..., 1], R1[..., 2]
    w2, x2, y2, z2 = R2[..., 3], R2[..., 0], R2[..., 1], R2[..., 2]

    R_rel = np.array([
        w1*w2 + x1*x2 + y1*y2 + z1*z2,
        w1*x2 - x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 - y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 - z1*w2
    ])

    R_rel = R_rel / np.linalg.norm(R_rel)  # normalize
    w, x, y, z = R_rel
    theta = 2.0 * np.arctan2(np.sqrt(x*x + y*y + z*z), np.abs(w))

    return np.degrees(theta)


def translation_error(t1, t2):
    #TODO: calculate translation error
    t1 = np.squeeze(t1)
    t2 = np.squeeze(t2)
    t_error = np.linalg.norm(t1 - t2)

    return t_error

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    #TODO: visualize the camera pose
    points3D = np.vstack(points3D_df["XYZ"].values)
    rgb_t = np.vstack(points3D_df["RGB"].values)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    pcd.colors = o3d.utility.Vector3dVector(rgb_t.astype(np.float64) / 255.0)

    # 錐體頂點 (相機中心)
    apex = np.array([[0, 0, 0]])
    # 底面四個角（假設相機面在 z=1 平面）
    base = np.array([
        [-1, -0.75, 1],
        [1, -0.75, 1],
        [1, 0.75, 1],
        [-1, 0.75, 1]
    ]) * 0.1

    vertices = np.vstack((apex, base))
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 從相機中心連到底面四角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 底面框線
    ]

    camera_meshes = []
    trajectory_points = []
    for T in Camera2World_Transform_Matrixs:
        cam = o3d.geometry.LineSet()
        cam.points = o3d.utility.Vector3dVector(vertices)
        cam.lines = o3d.utility.Vector2iVector(lines)
        cam.paint_uniform_color([1, 0, 0])

        inv_T = np.linalg.inv(T)
        cam.transform(inv_T)
        t = inv_T[:3, 3]
        camera_meshes.append(cam)
        trajectory_points.append(t)

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(np.array(trajectory_points))
    trajectory.lines = o3d.utility.Vector2iVector(
        [[i, i+1] for i in range(len(trajectory_points)-1)]
    )
    trajectory.paint_uniform_color([0, 0, 1])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(trajectory)
    for mesh in camera_meshes:
        vis.add_geometry(mesh)
    vc = vis.get_view_control()
    vc.set_zoom(0.1)
    o3d.visualization.ViewControl.set_zoom(vc, 1)
    vis.run()

    vis.destroy_window()


    pass


if __name__ == "__main__":
    # Load data
    start = time.time()
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    points3D_df = points3D_df[points3D_df["POINT_ID"] != -1]
    images_df = images_df.sort_values(by="NAME", key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))
    valid_images_df = images_df[images_df["NAME"].str.contains("valid")]

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    IMAGE_ID_LIST = valid_images_df["IMAGE_ID"]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq, rotq_gt)
        t_error = translation_error(tvec, tvec_gt)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    R_median = np.median(rotation_error_list)
    t_median = np.median(translation_error_list)
    end = time.time()

    print(f"Median of Rotation Error: {R_median}\nMedian of Translation Error: {t_median}\nTime: {end - start}")

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        c2w = np.eye(4)
        R_est, _ = cv2.Rodrigues(r)
        c2w[:3, :3] = R_est
        c2w[:3, 3] = t.reshape(3)
        Camera2World_Transform_Matrixs.append(c2w)
    visualization(Camera2World_Transform_Matrixs, points3D_df)