#! /usr/bin/env python

import actionlib
import graspit_shape_completion.msg
import geometry_msgs.msg
import shape_msgs.msg
import rospy

import numpy as np
import mcubes
from experiments import reconstruction_3d_test
from utils import reconstruction_utils
import visualization.visualize as viz
import pcl


class ShapeCompletionServer(object):
    # create messages that are used to publish feedback/result
    _feedback = graspit_shape_completion.msg.CompleteMeshFeedback()
    _result = graspit_shape_completion.msg.CompleteMeshResult()

    def __init__(self, name):

        rospy.loginfo('Loading model.')

        weights_filepath = '/home/avinash/research/shape_completion/train/shape_completion_experiments/experiments/best_weights.h5'
        self.patch_size = 30
        self.model = reconstruction_3d_test.get_model()
        self.model.load_weights(weights_filepath)

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, graspit_shape_completion.msg.CompleteMeshAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        rospy.loginfo('Node Initialized')

    def execute_cb(self, goal):

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

        batch_x[0, :, :, :, :] = reconstruction_utils.build_test_from_pc_scaled(pc, self.patch_size)

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        pred = self.model._predict(batch_x)
        pred = pred.reshape(1, self.patch_size, 1, self.patch_size, self.patch_size)

        pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

        v, t = mcubes.marching_cubes(pred_as_b012c[0, :, :, :, 0], 0.5)
        #mcubes.export_mesh(v, t, 'results/' + 'model_' + str(1) + '.dae', 'model')
        #viz.visualize_batch_x(pred, 0, str(1), 'results/' + "pred_" + str(1))
        #viz.visualize_batch_x(batch_x, 0, str(1), 'results/' + "input_" + str(1))

        for i in range(len(v)):
            self._result.completed_mesh.vertices.append(geometry_msgs.msg.Point(v[i, 0], v[i, 1], v[i, 2]))
        for i in range(len(t)):
            self._result.completed_mesh.triangles.append(shape_msgs.msg.MeshTriangle((t[i, 0], t[i, 1], t[i, 2])))

        self._as.set_succeeded(self._result)

if __name__ == '__main__':
    rospy.init_node('shape_completion_server')
    ShapeCompletionServer('/complete_mesh')
    rospy.spin()