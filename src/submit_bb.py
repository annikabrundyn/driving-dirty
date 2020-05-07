"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
from pytorch_lightning import Trainer

# Import model classes
from src.autoencoder.autoencoder import BasicAE
from src.roadmap_model.roadmap_pretrain_ae import RoadMap
from src.roadmap_model.roadmap_bce_v2 import RoadMapBCE
from src.bounding_box_model.bb_coord_reg.bb_MLP import Boxes
from src.bounding_box_model.spatial_bb.spatial_model import BBSpatialModel
from src.bounding_box_model.spatial_bb.spatial_w_rm import BBSpatialRoadMap
from src.bounding_box_model.fast_rcnn.bb_fast_rcnn import FasterRCNN

from test_tube import HyperOptArgumentParser, SlurmCluster
import os, sys

MODEL_NAMES = {
    'basic_ae': BasicAE,
    'roadmap_mse': RoadMap,
    'roadmap_bce': RoadMapBCE,
    'bb_reg': Boxes,
    'spatial_bb': BBSpatialModel,
    'spatial_rm': BBSpatialRoadMap,
    'faster_rcnn': FasterRCNN
}

def main_local(hparams):
    main(hparams, None)

def main(hparams, cluster):
    # init module
    MODEL = MODEL_NAMES[hparams.model]
    model = MODEL(hparams)

    path = os.path.join(hparams.logs_save_path, hparams.tt_name)
    hparams.default_root_dir = path

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(hparams)
    # trainer = Trainer(
    #     default_root_dir=hparams.default_root_dir,
    #     gpus=hparams.gpus,
    #     precision=hparams.precision
    # )
    trainer.fit(model)


def run_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(hyperparam_optimizer=hyperparams,
                           log_path=hyperparams.logs_save_path)

    # email results if your hpc supports it
    cluster.notify_job_status(email='ab8690@nyu.edu', on_done=True, on_fail=True)
    # any modules for code to run in env
    cluster.add_command(f'source activate {hyperparams.conda_env}')
    # pick the gpu resources
    cluster.per_experiment_nb_gpus = hyperparams.gpus
    cluster.per_experiment_nb_cpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.gpu_type = 'k80'
    # cluster.job_time = '20:00:00'
    cluster.job_time = '24:00:00'
    cluster.minutes_to_checkpoint_before_walltime = 5
    cluster.memory_mb_per_node = 30000
    # come up with a short exp name
    job_display_name = hyperparams.tt_name.split('_')[0]
    job_display_name = job_display_name[0:4]
    # optimize across all gpus
    print('submitting jobs...')
    cluster.optimize_parallel_cluster_gpu(main,
                                          nb_trials=hyperparams.nb_hopt_trials,
                                          job_name=job_display_name)

if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]

    parser = HyperOptArgumentParser(add_help=False, strategy='grid_search')
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, default='faster_rcnn')

    (temp_args, arr) = parser.parse_known_args()
    model_name = temp_args.model
    MODEL_CLASS = MODEL_NAMES[model_name]

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = MODEL_CLASS.add_model_specific_args(parser)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--conda_env', type=str, default='driving-dirty')
    parser.add_argument('--on_cluster', default=True, action='store_true')
    parser.add_argument('-n', '--tt_name', default='frcnn_newckpt')
    parser.add_argument('-d', '--tt_description', default='pretrained ae for feature extraction')
    parser.add_argument('--logs_save_path', default='/scratch/ab8690/logs')
    parser.add_argument('--single_run', dest='single_run', action='store_true')
    parser.add_argument('--nb_hopt_trials', default=3, type=int)

    # parse params
    hparams = parser.parse_args()
    if hparams.on_cluster and not hparams.single_run:
        run_on_cluster(hparams)
    else:
        main_local(hparams)
