"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to optimize the SE(3) pose (in the robot's base frame) of the 
specified camera. There are two pointclouds that we care about: pcd_real and pcd_sim. pcd_real is the measured 
pointcloud from the camera (found by projecting the depth points with the camera's intrinsic matrix). pcd_sim is a 
rendered pointcloud from ManiSkill. To get the rendering, a simulation is run. The robot is set to the measured joint 
angles. The camera is then moved to the specified camera pose, and lastly the pointcloud is rendered. The cost function 
is the chamfer distance between pcd_real and pcd_sim.

Notes: 
1. A preproccessing step is performed on the measured pointclouds to remove points that aren't part of the robot. To
    do so, the mask of the robot is generated at each timestep and used to mask the pointcloud. See 
    ImageUtils.get_sam_mask for details.
2. If --visualize is set, a viser server is started. The server will show the measured pointcloud, and the best to date
    (lowest total cost) rendered pointcloud.
3. if --visualize-robot-masks is set, the masks will be saved in the same directory as the output yaml file with the
    camera_name__robot_mask__timestep.png file format



The pseudo-code is as follows:

inputs:
- rgbds: array of RGBD images
- all_joint_angles: array of joint angles from the the demonstration. Should match the recorded time steps of the rgbds.
- N: the number of timesteps to sample from the demonstration for the cost function
- rgb_to_pcd(rgbd): converts the RGBD image to a robot only pointcloud
- S: number of poses in the CMA-ES population
- render_sim_pointcloud(pose, joint_angle): renders the pointcloud from the simulation at the given pose and joint angle
- compute_chamfer_distance(pcd_real, pcd_sim): computes the chamfer distance between the two pointclouds

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

pcd_reals = [rgb_to_pcd(rgbd) for rgbd in rgbds]

until convergence:
    sample poses pi, ..., pN from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle in joint_angles:
            pcd_sim = render_sim_pointcloud(pose, joint_angle)
            cost = compute_chamfer_distance(pcd_real, pcd_sim)
            total_cost += cost
        total_costs[i] = total_cost
    update CMA-ES with total_costs as fitness values

output:
- estimated pose
"""