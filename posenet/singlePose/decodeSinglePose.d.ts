import * as tf from '@tensorflow/tfjs';
import { OutputStride } from '../mobilenet';
import { Pose } from '../types';
export declare function decodeSinglePose(heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D, outputStride: OutputStride): Promise<Pose>;
