from scipy.spatial.transform import Rotation as R
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

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 進行 kNN 比對（每個 query 找兩個最相似的 model descriptor）
    # matches = bf.knnMatch(desc_query, desc_model, k=2)

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

    # 使用 RANSAC 求解PnP
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=8.0,
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