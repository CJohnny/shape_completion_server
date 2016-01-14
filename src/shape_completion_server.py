#! /usr/bin/env python

import actionlib
import graspit_shape_completion.msg
import geometry_msgs.msg
import shape_msgs.msg
import rospy

import argparse
import subprocess
import numpy as np
from scipy.ndimage import map_coordinates
import mcubes
from experiments import reconstruction_3d_test
from utils import reconstruction_utils
from off_utils import off_handler
from shape_completion_server.srv import *
import time

class ShapeCompletionServer(object):
    # create messages that are used to publish feedback/result
    _feedback = graspit_shape_completion.msg.CompleteMeshFeedback()
    _result = graspit_shape_completion.msg.CompleteMeshResult()

    def __init__(self, name, smooth_mesh):

        rospy.loginfo('Loading model.')

        self.weights_filepath = '/home/avinash/research/shape_completion/train/shape_completion_experiments/experiments/ycb_all_shrec_best_weights.h5'
        self.patch_size = 30
        self.model = reconstruction_3d_test.get_model()
        self.model.load_weights(self.weights_filepath)
        self.smooth_mesh = smooth_mesh

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, graspit_shape_completion.msg.CompleteMeshAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        self.completion_info_service = rospy.Service('completion_info', CompletionInfo, self.completion_info_cb)

        rospy.loginfo('Node Initialized')

    def completion_info_cb(self, req):
        return CompletionInfoResponse(self.weights_filepath)



    def execute_cb(self, goal):

        start_time = time.time()
        self._feedback = graspit_shape_completion.msg.CompleteMeshFeedback()
        self._result = graspit_shape_completion.msg.CompleteMeshResult()

        rospy.loginfo('Received Msg')
        single_view_pointcloud_filepath = '/srv/data/shape_completion_data/test_1/pcd_8_310_143723.pcd'

        point_array = np.asarray(goal.partial_mesh.vertices)
        pc = np.zeros((len(point_array), 3), np.float32)
        for i in range(len(point_array)):
            pc[i][0] = point_array[i].x
            pc[i][1] = point_array[i].y
            pc[i][2] = point_array[i].z

        batch_x = np.zeros((1, self.patch_size, self.patch_size, self.patch_size, 1), dtype=np.float32)

        batch_x[0, :, :, :, :], voxel_resolution, offset = reconstruction_utils.build_test_from_pc_scaled(pc, self.patch_size)

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        pred = self.model._predict(batch_x)
        pred = pred.reshape(1, self.patch_size, 1, self.patch_size, self.patch_size)

        pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        mask = reconstruction_utils.get_occluded_voxel_grid(batch_x[0, :, :, :, 0])
        completed_region = pred_as_b012c[0, :, :, :, 0] * mask

        scale = 4
        high_res_voxel_grid, voxel_resolution, offset = reconstruction_utils.build_high_res_voxel_grid(pc, scale, self.patch_size)
        indices = np.mgrid[0:scale*self.patch_size:1, 0:scale*self.patch_size:1, 0:scale*self.patch_size:1]
        scaled_completed_region = map_coordinates(completed_region, indices/scale, order = 0, mode = 'constant', cval=0.0)

        output = high_res_voxel_grid[:, :, :, 0] + scaled_completed_region
        #output = batch_x[0, :, :, :, 0] + completed_region

        v, t = mcubes.marching_cubes(output, 0.25)
        if self.smooth_mesh:
            coarse_mesh = '/srv/data/temp/' + 'coarse.dae'
            smooth_mesh = '/srv/data/temp/' + 'smooth.off'
            script_file = '/srv/data/temp/' + 'poisson_remesh.mlx'
            mcubes.export_mesh(v, t, coarse_mesh, 'model')
            cmd_string = 'meshlabserver -i ' + coarse_mesh
            cmd_string = cmd_string + ' -o ' + smooth_mesh
            cmd_string = cmd_string + ' -s ' + script_file
            process = subprocess.call(cmd_string, shell=True)
            off = off_handler.OffHandler()
            off.read(smooth_mesh)
            v = off.vertices
            t = off.faces

        #viz.visualize_batch_x(pred, 0, str(1), 'results/' + "pred_" + str(1))
        #viz.visualize_batch_x(batch_x, 0, str(1), 'results/' + "input_" + str(1))

        v *= voxel_resolution
        v += offset

        for i in range(len(v)):
            self._result.completed_mesh.vertices.append(geometry_msgs.msg.Point(v[i, 0], v[i, 1], v[i, 2]))
        for i in range(len(t)):
            self._result.completed_mesh.triangles.append(shape_msgs.msg.MeshTriangle((t[i, 0], t[i, 1], t[i, 2])))

        end_time = time.time()
        self._result.completion_time = int(1000*(end_time - start_time))
        self._as.set_succeeded(self._result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Completes the 3D shape of input point clouds')
    parser.add_argument('--smooth',
                        default=0,
                        help='1: smooths the reconstructed mesh. 0: no smoothing is applied',
                        type=int)
    params = parser.parse_args()

    if params.smooth:
        rospy.loginfo('With Smoothing')
    else:
        rospy.loginfo('Without Smoothing')

    rospy.init_node('shape_completion_server')
    ShapeCompletionServer('/complete_mesh', params.smooth)
    rospy.spin()